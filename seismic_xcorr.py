# LICENSE
#
# This file is part of psysmon_seismic.
#
'''
SeisMIC Correlator Looper Child Node for pSysmon.
'''

import os
import shutil
import logging
import numpy as np
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from obspy.core import Stream, UTCDateTime
import psysmon
import psysmon.core.packageNodes as package_nodes
from psysmon.core.preferences_manager import (
    SingleChoicePrefItem,
    FloatSpinPrefItem,
    IntegerSpinPrefItem,
    CheckBoxPrefItem,
    DirBrowsePrefItem
)

if psysmon.wx_available:
    import psysmon.gui.dialog.pref_listbook as psy_lb

from psysmon_seismic.client import PsysmonStoreClient, psysmon_to_obspy_inventory
from psysmon_seismic.utils import get_seismic_dirs, build_correlator_config

logger = logging.getLogger(__name__)


def _vectorized_clip(A: np.ndarray, args: dict, params: dict) -> np.ndarray:
    ''' Vectorized in-place clipping using C-level np.clip broadcasting. '''
    stds = np.nanstd(A, axis=1, keepdims=True)
    ts = args['std_factor'] * stds
    np.clip(A, -ts, ts, out=A)
    return A


def _vectorized_detrendqr(data: np.ndarray) -> np.ndarray:
    ''' Vectorized 2D detrending via BLAS GEMM matrix multiplications. '''
    npts = data.shape[-1]
    X = np.ones((npts, 2), dtype=np.float32)
    X[:, 0] = np.arange(0, npts, dtype=np.float32) / npts
    Q, R = np.linalg.qr(X)
    rq = np.dot(np.linalg.inv(R), Q.transpose())
    if data.ndim == 1:
        coeff = np.dot(rq, data)
        data = data - np.dot(X, coeff)
    elif data.ndim == 2:
        coeff = np.dot(data, rq.T)
        data -= np.dot(coeff, X.T)
    else:
        raise ValueError('data must be 1D or 2D')
    return data


def _vectorized_FDfilter(B: np.ndarray, args: dict, params: dict) -> np.ndarray:
    ''' Zero-allocation frequency domain filtering via direct broadcasting. '''
    import obspy.signal as osignal
    from copy import deepcopy
    args = deepcopy(args)
    args.update({'freqs': params['freqs']})
    tap = osignal.invsim.cosine_taper(B.shape[1], **args)
    B *= tap
    return B


def _vectorized_pxcorr_matrix(self, A: np.ndarray):
    ''' High-performance vectorized cross-correlation matrix computation.
    
    Batches Fourier transforms across 2D frequency-domain matrices, vectorizes
    energy norm calculations, and eliminates Python combination loops.
    '''
    from mpi4py import MPI
    import seismic.correlate.preprocessing_td as pptd
    from seismic.utils.fetch_func_from_str import func_from_str

    ntrc = A.shape[0]
    ind = self._get_row_index_per_core(A)
    self.logger.info(
        "Core %d processing %d matrix rows" % (self.rank, np.sum(ind))
    )
    try:
        self.logger.debug(
            "Core {:d} working on indices {:d}-{:d}".format(
                self.rank, *list(np.where(ind)[0][[0, -1]])
            )
        )
    except IndexError:
        self.logger.debug(
            "Core {:d} working on no indices".format(self.rank)
        )

    corr_args = self.options["corr_args"]
    params = {}
    for key in list(corr_args.keys()):
        if "Processing" not in key:
            params.update({key: corr_args[key]})
    params["sampling_rate"] = self.sampling_rate

    np.nan_to_num(A, copy=False)

    for proc in corr_args["TDpreProcessing"]:
        self.logger.info("Running %s" % proc["function"])
        func = func_from_str(proc["function"])
        A[ind, :] = func(A[ind, :], proc["args"], params)
        self.logger.info("Finished %s" % proc["function"])

    A = pptd.zeroPadding(A, {"type": "avoidWrapFastLen"}, params)

    self.logger.info("Core %d performing FFT." % self.rank)
    zmsize = A.shape
    fftsize = zmsize[1] // 2 + 1
    B = np.zeros((ntrc, fftsize), dtype=np.csingle)
    B[ind, :] = np.fft.rfft(A[ind, :], axis=1)

    freqs = np.fft.rfftfreq(zmsize[1], 1.0 / self.sampling_rate)
    params.update({"freqs": freqs})

    for proc in corr_args["FDpreProcessing"]:
        self.logger.info("Running %s" % proc["function"])
        func = func_from_str(proc["function"])
        B[ind, :] = func(B[ind, :], proc["args"], params)
        self.logger.info("Finished %s" % proc["function"])

    B = np.ascontiguousarray(B, dtype=np.complex64)
    self.comm.Allreduce(MPI.IN_PLACE, [B, MPI.COMPLEX], op=MPI.SUM)

    self.logger.info("Core %d performing correlation." % self.rank)
    csize = len(self.options["combinations"])
    irfftsize = (fftsize - 1) * 2
    sampleToSave = int(
        np.ceil(corr_args["lengthToSave"] * self.sampling_rate)
    )
    C = np.zeros((csize, sampleToSave * 2 + 1), dtype=np.float32)

    pmap = np.arange(csize) * self.psize / csize
    pmap = pmap.astype(np.int32)
    ind = pmap == self.rank
    ind = np.arange(csize)[ind]
    startlags = np.zeros(csize, dtype=np.float32)
    self.logger.info(
        "Core %d correlating %d matrix rows" % (self.rank, ind.size)
    )

    if ind.size > 0:
        c0 = np.array([self.options["combinations"][ii][0] for ii in ind], dtype=np.int32)
        c1 = np.array([self.options["combinations"][ii][1] for ii in ind], dtype=np.int32)

        starttimes = np.array(self.options["starttime"], dtype=np.float64)
        offsets = starttimes[c0] - starttimes[c1]

        if corr_args["center_correlation"]:
            roffsets = np.zeros_like(offsets)
        else:
            roffsets = np.fix(offsets * self.sampling_rate) / self.sampling_rate
        frac_offsets = offsets - roffsets

        if corr_args["normalize_correlation"]:
            trace_energy = 2.0 * np.sum(B.real**2 + B.imag**2, axis=1) - B[:, 0].real**2
            trace_norms = np.sqrt(np.maximum(trace_energy, 0.0)).astype(np.float32)
            norms = (trace_norms[c0] * trace_norms[c1] / irfftsize).astype(np.float32)
            np.maximum(norms, 1e-12, out=norms)
        else:
            norms = np.ones(len(ind), dtype=np.float32)

        if np.all(np.isclose(frac_offsets, 0.0)):
            M = B[c0, :].conj() * B[c1, :]
        else:
            phase = np.exp(1j * (2.0 * np.pi) * frac_offsets[:, None] * freqs[None, :])
            M = (B[c0, :].conj() * B[c1, :]) * phase

        tmp = np.fft.irfft(M, n=irfftsize, axis=1).real.astype(np.float32)
        C[ind, :] = np.hstack([tmp[:, -sampleToSave:], tmp[:, :sampleToSave + 1]]) / norms[:, None]
        startlags[ind] = (-sampleToSave / self.sampling_rate - roffsets).astype(np.float32)

    self.logger.info(
        "Core %d finished correlation for %d combinations."
        % (self.rank, len(ind))
    )

    self.logger.debug("Core %d summing corr matrix." % self.rank)
    self.comm.Allreduce(MPI.IN_PLACE, [C, MPI.FLOAT], op=MPI.SUM)
    self.logger.debug("Core %d summing startlags." % self.rank)
    self.comm.Allreduce(MPI.IN_PLACE, [startlags, MPI.FLOAT], op=MPI.SUM)
    self.logger.info("Core %d shared results between cores." % self.rank)
    return (C, startlags)


def _apply_worker_correlator_optimizations():
    ''' Apply in-memory vectorized optimizations to SeisMIC modules in worker process. '''
    import seismic.correlate.correlate as sc_corr
    import seismic.correlate.preprocessing_td as pptd
    import seismic.correlate.preprocessing_fd as ppfd

    sc_corr.Correlator._pxcorr_matrix = _vectorized_pxcorr_matrix
    pptd.clip = _vectorized_clip
    pptd.detrendqr = _vectorized_detrendqr
    ppfd.FDfilter = _vectorized_FDfilter


@contextmanager
def _patch_correlator_optimizations():
    ''' Scoped context manager to inject vectorized optimizations into SeisMIC Correlator. '''
    import seismic.correlate.correlate as sc_corr
    import seismic.correlate.preprocessing_td as pptd
    import seismic.correlate.preprocessing_fd as ppfd

    orig_pxcorr = sc_corr.Correlator._pxcorr_matrix
    orig_clip = pptd.clip
    orig_detrendqr = pptd.detrendqr
    orig_fdfilter = ppfd.FDfilter

    sc_corr.Correlator._pxcorr_matrix = _vectorized_pxcorr_matrix
    pptd.clip = _vectorized_clip
    pptd.detrendqr = _vectorized_detrendqr
    ppfd.FDfilter = _vectorized_FDfilter
    try:
        yield
    finally:
        sc_corr.Correlator._pxcorr_matrix = orig_pxcorr
        pptd.clip = orig_clip
        pptd.detrendqr = orig_detrendqr
        ppfd.FDfilter = orig_fdfilter


def _parallel_correlator_worker(task_args):
    ''' Worker task executed in a parallel subprocess by ProcessPoolExecutor. '''
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    import seismic.correlate.preprocessing_stream as ppst
    from seismic.correlate.correlate import Correlator
    from psysmon_seismic.client import PsysmonStoreClient
    from psysmon_seismic.utils import build_correlator_config

    # Bypass redundant SeisMIC 24h stream detrending
    ppst.detrend_st = lambda s, *a, **kw: s

    # Inject high-performance vectorized operations
    _apply_worker_correlator_optimizations()

    (
        idx, c_start_ts, c_end_ts, chunk_len,
        proj_base, sds_subdir, pref_dict,
        stream_meta, obs_inventory, sub_stream
    ) = task_args

    corr_sub = f"corr_worker_{idx}"
    log_sub = f"log_{idx}"

    cfg = build_correlator_config(
        pref_dict,
        proj_dir=proj_base,
        sds_subdir=sds_subdir,
        corr_subdir=corr_sub,
        log_subdir=log_sub,
        stream_meta=stream_meta,
        read_start=c_start_ts,
        read_end=c_end_ts,
        read_len=chunk_len,
        read_inc=chunk_len,
    )

    sc = PsysmonStoreClient(
        None,
        proj_base,
        sds_subdir=sds_subdir,
        inventory=obs_inventory,
        stream=sub_stream
    )
    correlator = Correlator(options=cfg, store_client=sc)
    correlator.pxcorr()
    return corr_sub


def _merge_hdf5_worker_dirs(base_dir, worker_corr_dirs, final_corr_dir):
    ''' Fast copy-merge of worker HDF5 datasets into primary correlation databases. '''
    import h5py
    os.makedirs(final_corr_dir, exist_ok=True)
    for worker_dir in worker_corr_dirs:
        full_worker_path = os.path.join(base_dir, worker_dir)
        if not os.path.isdir(full_worker_path):
            continue
        for fname in os.listdir(full_worker_path):
            if not fname.endswith('.h5'):
                continue
            src_path = os.path.join(full_worker_path, fname)
            dst_path = os.path.join(final_corr_dir, fname)
            with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'a') as dst:
                def copy_dataset(name, obj):
                    if isinstance(obj, h5py.Dataset) and name not in dst:
                        src.copy(name, dst, name=name)
                src.visititems(copy_dataset)
        shutil.rmtree(full_worker_path, ignore_errors=True)


class SeismicXCorrNode(package_nodes.LooperCollectionChildNode):
    ''' SeisMIC Cross-Correlation and Autocorrelation Looper Child Node.

    Executes within TimeWindowLooper to compute dual-domain preprocessed
    cross-correlations or autocorrelations and accumulate them into HDF5 databases.
    '''
    name = 'SeisMIC Correlator'
    mode = 'looper child'
    category = 'Interferometry'
    tags = ['seismic', 'interferometry', 'cross-correlation', 'autocorrelation', 'looper child']

    def __init__(self, **kwargs):
        package_nodes.LooperCollectionChildNode.__init__(self, **kwargs)
        self.create_preferences()

    def create_preferences(self):
        ''' Create preference pages and groups for correlation and conditioning. '''
        pref_page = self.pref_manager.add_page('Correlation')
        
        corr_group = pref_page.add_group('Correlation Parameters')
        item = SingleChoicePrefItem(
            name='combination_method',
            label='combination method',
            value='autoComponents',
            limit=('autoComponents', 'betweenStations', 'allSimpleCombinations'),
            tool_tip=(
                'autoComponents: Autocorrelations for each station/channel.\n'
                'betweenStations: Cross-correlations between distinct stations.\n'
                'allSimpleCombinations: All pairwise combinations.'
            )
        )
        corr_group.add_item(item)

        item = FloatSpinPrefItem(
            name='corr_len',
            label='window length [s]',
            value=3600.0,
            limit=(0.1, 1e8),
            increment=60.0,
            digits=1,
            tool_tip='Length of individual correlation windows in seconds.'
        )
        corr_group.add_item(item)

        item = FloatSpinPrefItem(
            name='corr_inc',
            label='window increment [s]',
            value=3600.0,
            limit=(0.1, 1e8),
            increment=60.0,
            digits=1,
            tool_tip='Step increment between correlation windows in seconds.'
        )
        corr_group.add_item(item)

        item = FloatSpinPrefItem(
            name='max_lag',
            label='max lag [s]',
            value=2.0,
            limit=(0.01, 1e6),
            increment=0.1,
            digits=2,
            tool_tip='Maximum lag time for cross/autocorrelation functions in seconds.'
        )
        corr_group.add_item(item)

        # Preprocessing Page
        prep_page = self.pref_manager.add_page('Preprocessing')

        td_group = prep_page.add_group('Time Domain Conditioning')
        item = FloatSpinPrefItem(
            name='clip_factor',
            label='clipping factor',
            value=0.0,
            limit=(0.0, 1e6),
            increment=0.5,
            digits=1,
            tool_tip='Amplitude threshold clipping factor (0.0 to disable).'
        )
        td_group.add_item(item)

        item = SingleChoicePrefItem(
            name='td_norm',
            label='time normalization',
            value='none',
            limit=('none', 'onebit', 'running_mean', 'joint_norm'),
            tool_tip='Time-domain normalization method.'
        )
        td_group.add_item(item)

        fd_group = prep_page.add_group('Frequency Domain Conditioning')
        item = CheckBoxPrefItem(
            name='spectral_whitening',
            label='spectral whitening',
            value=True,
            tool_tip='Apply spectral whitening in frequency domain.'
        )
        fd_group.add_item(item)

        item = FloatSpinPrefItem(
            name='whiten_waterlevel',
            label='whitening waterlevel',
            value=0.0001,
            limit=(0.0, 100.0),
            increment=0.0001,
            digits=6,
            tool_tip='Waterlevel factor for spectral whitening division.'
        )
        fd_group.add_item(item)

        item = FloatSpinPrefItem(
            name='freq_min',
            label='filter min freq [Hz]',
            value=2.0,
            limit=(0.001, 10000.0),
            increment=0.5,
            digits=1,
            tool_tip='Lower corner frequency for bandpass conditioning.'
        )
        fd_group.add_item(item)

        item = FloatSpinPrefItem(
            name='freq_max',
            label='filter max freq [Hz]',
            value=25.0,
            limit=(0.001, 10000.0),
            increment=0.5,
            digits=1,
            tool_tip='Upper corner frequency for bandpass conditioning.'
        )
        fd_group.add_item(item)

        item = IntegerSpinPrefItem(
            name='filter_corners',
            label='filter corners',
            value=4,
            limit=(1, 10),
            tool_tip='Number of poles/corners for the bandpass filter.'
        )
        fd_group.add_item(item)

        par_group = pref_page.add_group('Parallel Processing')
        item = IntegerSpinPrefItem(
            name='parallel_workers',
            label='number of workers',
            value=4,
            limit=(1, 32),
            tool_tip=(
                'Number of parallel worker processes for temporal chunking.\n'
                'Set to 1 for sequential single-core execution.'
            )
        )
        par_group.add_item(item)

        log_group = pref_page.add_group('Logging')
        item = SingleChoicePrefItem(
            name='log_level',
            label='log level',
            value='WARNING',
            limit=('WARNING', 'INFO', 'DEBUG'),
            tool_tip=(
                'Logging verbosity for SeisMIC internal processing.\n'
                'WARNING: (Recommended) Suppresses verbose per-trace logs for faster execution.\n'
                'INFO: Detailed progress per correlation window.\n'
                'DEBUG: Full diagnostic logging.'
            )
        )
        log_group.add_item(item)

        # Plots page
        plot_page = self.pref_manager.add_page('Plots')
        out_group = plot_page.add_group('Plot Output')
        item = DirBrowsePrefItem(
            name='plot_dir',
            label='plot output directory',
            value='',
            tool_tip=(
                'Custom directory to save cross-correlation plots.\n'
                'Leave empty to auto-save to <process_dir>/plots/xcorr/.'
            )
        )
        out_group.add_item(item)

        item = SingleChoicePrefItem(
            name='plot_scope',
            label='plot scope',
            value='Cumulative Summary',
            limit=('Cumulative Summary', 'Daily (Per Iteration)', 'Both'),
            tool_tip=(
                'Cumulative Summary: Generates/updates a single master figure from the HDF5 database\n'
                '                    displaying the full timeline from Day 1 to the current day.\n'
                'Daily (Per Iteration): Generates separate figures for each 24h looper iteration (e.g. hourly sub-windows).\n'
                'Both: Generates both the per-iteration daily figure and updates the master summary.'
            )
        )
        out_group.add_item(item)

        item = SingleChoicePrefItem(
            name='plot_height_mode',
            label='height scaling mode',
            value='Full Looper Period (Progressive Fill)',
            limit=(
                'Full Looper Period (Progressive Fill)',
                'Current Accumulated Days (Elastic Growth)',
            ),
            tool_tip=(
                'Full Looper Period (Progressive Fill): Pre-allocates canvas height and locks y-axis to the full\n'
                '                                      planned looper period with constant pixel-per-day resolution.\n'
                'Current Accumulated Days (Elastic Growth): Dynamically grows canvas height with data processed so far,\n'
                '                                           leaving zero whitespace at the bottom.'
            )
        )
        out_group.add_item(item)

        item = SingleChoicePrefItem(
            name='plot_format',
            label='image format',
            value='png',
            limit=('png', 'pdf', 'svg', 'jpg'),
            tool_tip='File format for saved figures.'
        )
        out_group.add_item(item)

        item = IntegerSpinPrefItem(
            name='plot_dpi',
            label='resolution [dpi]',
            value=150,
            limit=(72, 600),
            tool_tip='Resolution in dots per inch for saved images.'
        )
        out_group.add_item(item)

        plots_group = plot_page.add_group('Available Plots')
        item = CheckBoxPrefItem(
            name='plot_heatmap',
            label='Cross-correlation temporal heatmap',
            value=False,
            tool_tip='2D image of lag time vs correlation sub-windows or calendar time.'
        )
        plots_group.add_item(item)

        item = CheckBoxPrefItem(
            name='plot_section',
            label='Cross-correlation wiggle record section',
            value=False,
            tool_tip='Wiggle-trace record section of correlations vs inter-station distance or time.'
        )
        plots_group.add_item(item)

        item = CheckBoxPrefItem(
            name='plot_stack',
            label='Stacked correlation waveform',
            value=False,
            tool_tip='Individual stacked cross-correlation waveform per station pair.'
        )
        plots_group.add_item(item)

    def edit(self):
        ''' Show the node edit dialog. '''
        if psysmon.wx_available:
            dlg = psy_lb.ListbookPrefDialog(preferences=self.pref_manager)
            dlg.ShowModal()
            dlg.Destroy()

    def execute(self, stream, process_limits=None, origin_resource=None, output_dir=None, **kwargs):
        ''' Execute cross/auto-correlation on the provided stream.

        Parameters
        ----------
        stream : :class:`obspy.core.Stream`
            The continuous data stream passed from the looper.
        process_limits : tuple of UTCDateTime, optional
            (start_time, end_time) of the current processing slice.
        origin_resource : optional
            Data source identifier.
        output_dir : str, optional
            Output process directory provided by TimeWindowLooper. If specified,
            all correlation databases, logs, and staging files will be saved into
            <output_dir>/seismic/.
        '''
        if not stream or len(stream) == 0:
            self.logger.warning("SeismicXCorrNode: Received empty stream, skipping.")
            return

        # Check if stream contains unstacked constituent mini-array channels
        unstacked_channels = [
            tr.id for tr in stream
            if tr.stats.location in ('A', 'B', 'C') or (len(tr.stats.station) > 4 and tr.stats.station[-1] in ('A', 'B', 'C'))
        ]
        if unstacked_channels:
            self.logger.warning(
                "SeismicXCorrNode: Incoming stream contains raw, unstacked station channels (%s). "
                "Proceeding to correlate raw traces directly without mini-array stacking.",
                unstacked_channels
            )

        # Emit notice regarding bypassed SeisMIC detrending
        self.logger.info(
            "SeismicXCorrNode: Redundant 24h stream detrending in SeisMIC is bypassed for performance. "
            "Ensure incoming waveforms have been detrended in the pSysmon processing stack prior to correlation."
        )

        # Resolve directories: use output_dir from TimeWindowLooper if available,
        # otherwise fall back to project data directory.
        if output_dir:
            target_base = output_dir
            self.logger.info("SeismicXCorrNode: Using looper process output directory: %s", target_base)
        else:
            target_base = getattr(self.project, 'dataDir', None) or getattr(self.project, 'data_directory', None)
            if not target_base:
                target_base = os.path.abspath('./psysmon_data')
            self.logger.info("SeismicXCorrNode: Standalone run; using project data directory: %s", target_base)

        seismic_dirs = get_seismic_dirs(target_base)
        proj_base = seismic_dirs['base']
        sds_subdir = 'sds_stage'
        corr_subdir = 'corr'
        log_subdir = 'log'
        sds_stage = os.path.join(proj_base, sds_subdir)
        os.makedirs(sds_stage, exist_ok=True)

        parallel_workers = int(self.pref_manager.get_value('parallel_workers') or 1)
        corr_inc = float(self.pref_manager.get_value('corr_inc') or 3600.0)

        t_start = min(tr.stats.starttime for tr in stream)
        t_end = max(tr.stats.endtime for tr in stream)
        total_duration = t_end - t_start
        total_increments = int(np.ceil(total_duration / corr_inc)) if corr_inc > 0 else 1

        import seismic.correlate.preprocessing_stream as ppst
        from seismic.correlate.correlate import Correlator

        if parallel_workers > 1 and total_increments >= 2:
            n_workers = min(parallel_workers, total_increments)
            self.logger.info(
                "SeismicXCorrNode: Executing parallel correlation across %d worker processes (%d total increments)...",
                n_workers, total_increments
            )
            # Extract stream metadata and inventory once
            net = stream[0].stats.network or 'XX'
            stations = sorted(list(set(tr.stats.station for tr in stream)))
            fs = stream[0].stats.sampling_rate
            stream_meta = (net, stations, fs, t_start, t_end)

            obs_inventory = None
            if self.project and hasattr(self.project, 'geometry_inventory'):
                try:
                    obs_inventory = psysmon_to_obspy_inventory(self.project.geometry_inventory)
                except Exception:
                    obs_inventory = None

            pref_dict = {
                'combination_method': self.pref_manager.get_value('combination_method'),
                'corr_len': self.pref_manager.get_value('corr_len'),
                'corr_inc': self.pref_manager.get_value('corr_inc'),
                'max_lag': self.pref_manager.get_value('max_lag'),
                'clip_factor': self.pref_manager.get_value('clip_factor'),
                'td_norm': self.pref_manager.get_value('td_norm'),
                'spectral_whitening': self.pref_manager.get_value('spectral_whitening'),
                'whiten_waterlevel': self.pref_manager.get_value('whiten_waterlevel'),
                'freq_min': self.pref_manager.get_value('freq_min'),
                'freq_max': self.pref_manager.get_value('freq_max'),
                'filter_corners': self.pref_manager.get_value('filter_corners'),
                'log_level': self.pref_manager.get_value('log_level'),
            }

            # Distribute increments among workers
            inc_splits = np.array_split(np.arange(total_increments), n_workers)
            tasks = []
            for idx, inc_chunk in enumerate(inc_splits):
                if len(inc_chunk) == 0:
                    continue
                c_start = t_start + float(inc_chunk[0] * corr_inc)
                c_end = t_start + float((inc_chunk[-1] + 1) * corr_inc)
                c_dur = c_end - c_start
                sub_st = stream.slice(c_start, c_end)
                tasks.append((
                    idx, c_start, c_end, c_dur,
                    proj_base, sds_subdir, pref_dict,
                    stream_meta, obs_inventory, sub_st
                ))

            with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
                worker_dirs = list(pool.map(_parallel_correlator_worker, tasks))

            _merge_hdf5_worker_dirs(proj_base, worker_dirs, seismic_dirs['corr'])
            self.logger.info("SeismicXCorrNode: Parallel correlation complete. Merged outputs in %s", seismic_dirs['corr'])

        else:
            # Sequential execution path (with in-memory client and bypassed detrending)
            orig_detrend_st = ppst.detrend_st
            ppst.detrend_st = lambda s, *a, **kw: s
            try:
                with _patch_correlator_optimizations():
                    config = build_correlator_config(
                        self.pref_manager,
                        proj_dir=proj_base,
                        sds_subdir=sds_subdir,
                        corr_subdir=corr_subdir,
                        log_subdir=log_subdir,
                        stream=stream
                    )
                    self.logger.info("SeismicXCorrNode: Starting SeisMIC Correlator (single process, method: %s)...", config['co']['combination_method'])
                    sc = PsysmonStoreClient(self.project, proj_base, sds_subdir=sds_subdir, stream=stream)
                    correlator = Correlator(options=config, store_client=sc)
                    correlator.pxcorr()
                    self.logger.info("SeismicXCorrNode: Correlator finished. Output databases in %s", seismic_dirs['corr'])
            finally:
                ppst.detrend_st = orig_detrend_st

        # Generate plots according to preferences
        self._generate_xcorr_plots(seismic_dirs['corr'], target_base, process_limits)

        # Clean up empty staging directory
        shutil.rmtree(sds_stage, ignore_errors=True)

    def _generate_xcorr_plots(self, corr_dir, target_base, process_limits):
        ''' Headless generation of cross-correlation plots based on user preferences. '''
        try:
            plot_heatmap = bool(self.pref_manager.get_value('plot_heatmap'))
            plot_section = bool(self.pref_manager.get_value('plot_section'))
            plot_stack = bool(self.pref_manager.get_value('plot_stack'))
            if not (plot_heatmap or plot_section or plot_stack):
                return

            user_plot_dir = str(self.pref_manager.get_value('plot_dir') or '').strip()
            plot_dir = user_plot_dir if user_plot_dir else os.path.join(target_base, 'plots', 'xcorr')
            os.makedirs(plot_dir, exist_ok=True)

            fmt = str(self.pref_manager.get_value('plot_format') or 'png').strip().lower()
            dpi = int(self.pref_manager.get_value('plot_dpi') or 150)
            plot_scope = str(self.pref_manager.get_value('plot_scope') or 'Cumulative Summary').strip()
            plot_height_mode = str(self.pref_manager.get_value('plot_height_mode') or 'Full Looper Period (Progressive Fill)').strip()
            max_lag = float(self.pref_manager.get_value('max_lag') or 5.0)
            corr_inc = float(self.pref_manager.get_value('corr_inc') or 3600.0)

            # Non-overlapping scaling factor: SeisMIC internally multiplies scalingfactor by 1e5
            # To maintain clean lane spacing with zero trace collision: (0.40 * corr_inc) / 1e5
            daily_scaling = max(0.002, (0.40 * corr_inc) / 1e5)

            # Discover planned looper timespan
            looper_start = None
            looper_end = None
            parent_looper = getattr(self, 'parent', None)
            if parent_looper and hasattr(parent_looper, 'pref_manager'):
                try:
                    looper_start = parent_looper.pref_manager.get_value('start_time')
                    looper_end = parent_looper.pref_manager.get_value('end_time')
                except Exception:
                    pass

            if not (looper_start and looper_end):
                meta_candidates = [
                    os.path.join(target_base, 'execution_metadata.json'),
                    os.path.join(os.path.dirname(target_base), 'execution_metadata.json')
                ]
                for meta_path in meta_candidates:
                    if os.path.exists(meta_path):
                        try:
                            import json
                            from obspy import UTCDateTime
                            with open(meta_path, 'r') as fp:
                                meta = json.load(fp)
                            node_settings = meta.get('node_settings', {})
                            for k, v in node_settings.items():
                                if 'time_window_looper' in k:
                                    if 'start_time' in v:
                                        looper_start = UTCDateTime(v['start_time'])
                                    if 'end_time' in v:
                                        looper_end = UTCDateTime(v['end_time'])
                                    break
                            if looper_start and looper_end:
                                break
                        except Exception:
                            pass

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from seismic.db.corr_hdf5 import CorrelationDataBase
            from seismic.correlate.stream import CorrStream

            if not os.path.isdir(corr_dir):
                return

            h5_files = [os.path.join(corr_dir, f) for f in os.listdir(corr_dir) if f.endswith('.h5')]
            for h5_path in h5_files:
                fname = os.path.basename(h5_path)
                comb_name = fname[:-3] if fname.endswith('.h5') else fname
                try:
                    with CorrelationDataBase(h5_path, mode='r') as cdb:
                        tags = [k for k in cdb.keys() if k not in ('co',)]
                        if not tags:
                            continue
                        tag = 'subdivision' if 'subdivision' in tags else ('recombined' if 'recombined' in tags else tags[0])
                        cst = cdb.get_data('*', '*', '*', '*', tag)
                        if not cst or cst.count() == 0:
                            continue

                        # Identify trace timestamp extents
                        cst_starts = [tr.stats.corr_start for tr in cst if hasattr(tr.stats, 'corr_start') and tr.stats.corr_start]
                        cst_ends = [tr.stats.corr_end for tr in cst if hasattr(tr.stats, 'corr_end') and tr.stats.corr_end]
                        curr_start = min(cst_starts) if cst_starts else None
                        curr_end = max(cst_ends) if cst_ends else None

                        # Resolve cumulative plot height and y-axis limits
                        if 'Full Looper' in plot_height_mode and looper_start and looper_end and looper_end > looper_start:
                            effective_days = max(1.0, (looper_end - looper_start) / 86400.0)
                            cum_ylimits = (looper_end.datetime, looper_start.datetime)
                        else:
                            effective_days = max(1.0, (curr_end - curr_start) / 86400.0) if (curr_start and curr_end and curr_end > curr_start) else 1.0
                            cum_ylimits = (curr_end.datetime, curr_start.datetime) if (curr_start and curr_end) else None

                        cum_fig_height = float(np.clip(2.5 + (effective_days * 0.45), 6.0, 36.0))

                        # 1. Daily plots (if requested and process_limits available)
                        if plot_scope in ('Daily (Per Iteration)', 'Both') and process_limits:
                            t_st, t_et = process_limits
                            daily_cst = CorrStream([
                                tr for tr in cst
                                if hasattr(tr.stats, 'corr_start') and tr.stats.corr_start >= t_st and tr.stats.corr_start <= t_et
                            ])
                            dt_label = t_st.strftime('%Y%m%d')
                            if daily_cst.count() > 0:
                                if plot_heatmap:
                                    fig = plt.figure(figsize=(9, 6))
                                    ax = fig.gca()
                                    daily_cst.plot(timelimits=[-max_lag, max_lag], type='heatmap', cmap='seismic', vmin=-0.5, vmax=0.5, ax=ax)
                                    fig.savefig(os.path.join(plot_dir, f"xcorr_heatmap_{comb_name}_{dt_label}.{fmt}"), dpi=dpi, bbox_inches='tight')
                                    plt.close(fig)

                                if plot_section:
                                    fig = plt.figure(figsize=(9, 6))
                                    ax = fig.gca()
                                    daily_cst.plot(timelimits=[-max_lag, max_lag], type='section', scalingfactor=daily_scaling, linewidth=0.5, ax=ax)
                                    for line in ax.get_lines():
                                        line.set_color('black')
                                        line.set_antialiased(False)
                                    fig.savefig(os.path.join(plot_dir, f"xcorr_section_{comb_name}_{dt_label}.{fmt}"), dpi=dpi, bbox_inches='tight')
                                    plt.close(fig)

                                if plot_stack:
                                    fig = plt.figure(figsize=(9, 3))
                                    ax = fig.gca()
                                    stk = daily_cst.stack()
                                    if isinstance(stk, CorrStream) and stk.count() > 0:
                                        stk = stk[0]
                                    stk.plot(tlim=[-max_lag, max_lag], ax=ax)
                                    fig.savefig(os.path.join(plot_dir, f"xcorr_stack_{comb_name}_{dt_label}.{fmt}"), dpi=dpi, bbox_inches='tight')
                                    plt.close(fig)

                        # 2. Cumulative summary plots (if requested)
                        if plot_scope in ('Cumulative Summary', 'Both'):
                            if plot_heatmap:
                                fig = plt.figure(figsize=(9, cum_fig_height))
                                ax = fig.gca()
                                cst.plot(timelimits=[-max_lag, max_lag], ylimits=cum_ylimits, type='heatmap', cmap='seismic', vmin=-0.5, vmax=0.5, ax=ax)
                                fig.savefig(os.path.join(plot_dir, f"xcorr_heatmap_{comb_name}_cumulative.{fmt}"), dpi=dpi, bbox_inches='tight')
                                plt.close(fig)

                            if plot_section:
                                # Group traces by calendar day
                                days_dict = {}
                                for tr in cst:
                                    if hasattr(tr.stats, 'corr_start') and tr.stats.corr_start:
                                        day_key = tr.stats.corr_start.strftime('%Y-%m-%d')
                                        days_dict.setdefault(day_key, []).append(tr)

                                if len(days_dict) > 1 and len(cst) > 24:
                                    # Multi-day looper run: stack sub-windows per calendar day (1 high-SNR wiggle per day)
                                    daily_stacked_trs = []
                                    for day_key, trs in sorted(days_dict.items()):
                                        day_stk = CorrStream(trs).stack()[0]
                                        day_stk.stats.corr_start = trs[0].stats.corr_start
                                        if hasattr(trs[0].stats, 'corr_end'):
                                            day_stk.stats.corr_end = trs[-1].stats.corr_end
                                        daily_stacked_trs.append(day_stk)
                                    section_cst = CorrStream(daily_stacked_trs)
                                    # Trace spacing is 1 day (86400s); 0.40x factor leaves 60% whitespace gap
                                    section_scaling = (0.40 * 86400.0) / 1e5
                                    sec_fig_height = float(np.clip(3.0 + (len(daily_stacked_trs) * 0.35), 6.0, 48.0))
                                else:
                                    # Single day / short looper slice: plot sub-windows directly with 0.40x factor
                                    section_cst = cst
                                    section_scaling = daily_scaling
                                    sec_fig_height = cum_fig_height

                                fig = plt.figure(figsize=(9, sec_fig_height))
                                ax = fig.gca()
                                section_cst.plot(timelimits=[-max_lag, max_lag], ylimits=cum_ylimits, type='section', scalingfactor=section_scaling, linewidth=0.5, ax=ax)
                                for line in ax.get_lines():
                                    line.set_color('black')
                                    line.set_antialiased(False)
                                fig.savefig(os.path.join(plot_dir, f"xcorr_section_{comb_name}_cumulative.{fmt}"), dpi=dpi, bbox_inches='tight')
                                plt.close(fig)

                            if plot_stack:
                                fig = plt.figure(figsize=(9, 3))
                                ax = fig.gca()
                                stk = cst.stack()
                                if isinstance(stk, CorrStream) and stk.count() > 0:
                                    stk = stk[0]
                                stk.plot(tlim=[-max_lag, max_lag], ax=ax)
                                fig.savefig(os.path.join(plot_dir, f"xcorr_stack_{comb_name}_cumulative.{fmt}"), dpi=dpi, bbox_inches='tight')
                                plt.close(fig)

                except Exception as ex:
                    self.logger.warning("SeismicXCorrNode: Could not generate plots for %s: %s", fname, ex)
                finally:
                    plt.close('all')

            self.logger.info("SeismicXCorrNode: Plots generated successfully in %s", plot_dir)
        except Exception as e:
            self.logger.warning("SeismicXCorrNode: Plot generation failed: %s", e)
