# LICENSE
#
# This file is part of psysmon_seismic.
#
'''
Utility functions for psysmon_seismic.
'''

import os
import re
import numpy as np
from obspy.core.utcdatetime import UTCDateTime


def scnl_to_seed_id(scnl):
    ''' Convert a pSysmon SCNL tuple or dict to a SEED ID string. '''
    station, channel, network, location = scnl[0], scnl[1], scnl[2], scnl[3]
    if location is None or location == '--':
        location = ''
    return f"{network}.{station}.{location}.{channel}"


def seed_id_to_scnl(seed_id):
    ''' Convert a SEED ID string into a pSysmon SCNL tuple. '''
    parts = seed_id.split('.')
    if len(parts) == 4:
        network, station, location, channel = parts
    elif len(parts) == 3:
        network, station, channel = parts
        location = ''
    else:
        raise ValueError(f"Invalid SEED ID format: {seed_id}")
    return (station, channel, network, location)


def get_seismic_dirs(base_dir):
    ''' Ensure and return standard directories for SeisMIC data and results.

    If base_dir ends with 'seismic', uses base_dir directly.
    Otherwise, creates and uses base_dir/seismic.
    '''
    norm_base = os.path.normpath(os.path.abspath(base_dir))
    if os.path.basename(norm_base) == 'seismic':
        seismic_base = norm_base
    else:
        seismic_base = os.path.join(norm_base, 'seismic')

    dirs = {
        'base': seismic_base,
        'sds_stage': os.path.join(seismic_base, 'sds_stage'),
        'corr': os.path.join(seismic_base, 'corr'),
        'dv': os.path.join(seismic_base, 'dv'),
        'spatial': os.path.join(seismic_base, 'spatial'),
        'log': os.path.join(seismic_base, 'log'),
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs


def resolve_correlation_dir(input_path):
    ''' Resolve process base directory and correlation directory from user preference.

    Accepts:
      1. Direct 'corr' folder (e.g. .../time_window_looper/seismic/corr or .../corr)
      2. 'seismic' folder (e.g. .../time_window_looper/seismic)
      3. Looper process folder containing 'seismic/corr' or 'corr'

    Returns:
      (process_dir, corr_dir) or (None, None) if not found/invalid.
    '''
    if not input_path or not os.path.isdir(input_path):
        return None, None

    input_path = os.path.abspath(input_path)

    # Case 1: user selected the 'corr' folder directly
    if os.path.basename(input_path) == 'corr':
        corr_dir = input_path
        parent = os.path.dirname(input_path)
        proc_dir = os.path.dirname(parent) if os.path.basename(parent) == 'seismic' else parent
        return proc_dir, corr_dir

    # Case 2: user selected 'seismic' folder containing 'corr'
    if os.path.basename(input_path) == 'seismic':
        corr_dir = os.path.join(input_path, 'corr')
        proc_dir = os.path.dirname(input_path)
        if os.path.isdir(corr_dir):
            return proc_dir, corr_dir

    # Case 3: user selected process folder containing 'seismic/corr'
    seismic_corr = os.path.join(input_path, 'seismic', 'corr')
    if os.path.isdir(seismic_corr):
        return input_path, seismic_corr

    # Case 4: user selected process folder containing 'corr' directly
    direct_corr = os.path.join(input_path, 'corr')
    if os.path.isdir(direct_corr):
        return input_path, direct_corr

    return input_path, None


def resolve_dv_dir(input_path):
    ''' Resolve process base directory and dv directory from user preference.

    Accepts:
      1. Direct 'dv' folder (e.g. .../time_window_looper/seismic/dv or .../dv)
      2. 'seismic' folder (e.g. .../time_window_looper/seismic)
      3. Looper process folder containing 'seismic/dv' or 'dv'

    Returns:
      (process_dir, dv_dir) or (None, None) if not found/invalid.
    '''
    if not input_path or not os.path.isdir(input_path):
        return None, None

    input_path = os.path.abspath(input_path)

    # Case 1: user selected the 'dv' folder directly
    if os.path.basename(input_path) == 'dv':
        dv_dir = input_path
        parent = os.path.dirname(input_path)
        proc_dir = os.path.dirname(parent) if os.path.basename(parent) == 'seismic' else parent
        return proc_dir, dv_dir

    # Case 2: user selected 'seismic' folder containing 'dv'
    if os.path.basename(input_path) == 'seismic':
        dv_dir = os.path.join(input_path, 'dv')
        proc_dir = os.path.dirname(input_path)
        if os.path.isdir(dv_dir):
            return proc_dir, dv_dir

    # Case 3: user selected process folder containing 'seismic/dv'
    seismic_dv = os.path.join(input_path, 'seismic', 'dv')
    if os.path.isdir(seismic_dv):
        return input_path, seismic_dv

    # Case 4: user selected process folder containing 'dv' directly
    direct_dv = os.path.join(input_path, 'dv')
    if os.path.isdir(direct_dv):
        return input_path, direct_dv

    return input_path, None


def calculate_array_centroid(coordinates_list):
    ''' Calculate the arithmetic centroid of coordinates for a mini-array. '''
    if not coordinates_list:
        return (0.0, 0.0, 0.0)
    arr = np.array(coordinates_list, dtype=np.float64)
    centroid_lat = float(np.mean(arr[:, 0]))
    centroid_lon = float(np.mean(arr[:, 1]))
    centroid_elev = float(np.mean(arr[:, 2]))
    return (centroid_lat, centroid_lon, centroid_elev)


def build_correlator_config(
    pref_source,
    proj_dir,
    sds_subdir,
    corr_subdir,
    log_subdir='log',
    stream=None,
    stream_meta=None,
    read_start=None,
    read_end=None,
    read_len=None,
    read_inc=None,
):
    ''' Build a SeisMIC correlator configuration dictionary from pSysmon preferences or dict. '''
    get_val = pref_source.get_value if hasattr(pref_source, 'get_value') else pref_source.get
    comb_method = get_val('combination_method') or 'autoComponents'
    corr_len = float(get_val('corr_len') or 3600.0)
    corr_inc = float(get_val('corr_inc') or 3600.0)
    max_lag = float(get_val('max_lag') or 2.0)
    log_level = get_val('log_level') or 'WARNING'
    
    # Preprocessing
    clip_factor = get_val('clip_factor')
    td_norm = get_val('td_norm') or 'none'
    spectral_whitening = bool(get_val('spectral_whitening'))
    whiten_waterlevel = float(get_val('whiten_waterlevel') or 1e-4)
    freq_min = float(get_val('freq_min') or 2.0)
    freq_max = float(get_val('freq_max') or 25.0)

    # Time domain conditioning
    td_list = [
        {'function': 'seismic.correlate.preprocessing_td.taper', 'args': {'type': 'cosine_taper', 'p': 0.05}},
        {'function': 'seismic.correlate.preprocessing_td.detrend', 'args': {'type': 'linear'}},
    ]
    if clip_factor is not None and float(clip_factor) > 0:
        td_list.append({
            'function': 'seismic.correlate.preprocessing_td.clip',
            'args': {'factor': float(clip_factor)}
        })
    if td_norm == 'onebit':
        td_list.append({
            'function': 'seismic.correlate.preprocessing_td.signBitNormalization',
            'args': {}
        })
    elif td_norm == 'running_mean':
        td_list.append({
            'function': 'seismic.correlate.preprocessing_td.runningMeanNormalization',
            'args': {'wsize': 1.0}
        })
    elif td_norm == 'joint_norm':
        td_list.append({
            'function': 'seismic.correlate.preprocessing_td.joint_norm',
            'args': {}
        })

    # Frequency domain conditioning
    fd_list = []
    if spectral_whitening:
        fd_list.append({
            'function': 'seismic.correlate.preprocessing_fd.spectralWhitening',
            'args': {'joint_norm': False}
        })
    if freq_min > 0 and freq_max > freq_min:
        fd_list.append({
            'function': 'seismic.correlate.preprocessing_fd.FDfilter',
            'args': {'flimit': [max(0.5, freq_min - 0.5), freq_min, freq_max, freq_max + 2.0]}
        })

    # Extract stream info
    if stream and len(stream) > 0:
        net = stream[0].stats.network or 'XX'
        stations = sorted(list(set(tr.stats.station for tr in stream)))
        fs = stream[0].stats.sampling_rate
        t_start = min(tr.stats.starttime for tr in stream)
        t_end = max(tr.stats.endtime for tr in stream)
    elif stream_meta:
        net, stations, fs, t_start, t_end = stream_meta
    else:
        net = 'AP'
        stations = ['AP231', 'AP232', 'AP233']
        fs = 100.0
        t_start = UTCDateTime("2023-05-01T00:00:00")
        t_end = UTCDateTime("2023-05-03T00:00:00")

    # Determine read time bounds
    if read_start is not None and read_end is not None:
        r_start_str = UTCDateTime(read_start).strftime("%Y-%m-%d %H:%M:%S.0")
        r_end_str = UTCDateTime(read_end).strftime("%Y-%m-%d %H:%M:%S.0")
        r_len = int(read_len if read_len is not None else (UTCDateTime(read_end) - UTCDateTime(read_start)))
        r_inc = int(read_inc if read_inc is not None else r_len)
    else:
        # Round t_start down to day start and t_end up to day end
        day_start = UTCDateTime(year=t_start.year, julday=t_start.julday)
        t_end_adj = (t_end - 1e-3) if t_end > t_start else t_end
        day_end = UTCDateTime(year=t_end_adj.year, julday=t_end_adj.julday) + 86400.0
        r_start_str = day_start.strftime("%Y-%m-%d %H:%M:%S.0")
        r_end_str = day_end.strftime("%Y-%m-%d %H:%M:%S.0")
        r_len = 86400
        r_inc = 86400

    params = {
        'proj_dir': proj_dir,
        'log_subdir': log_subdir,
        'log_level': log_level,
        'fig_subdir': 'figures',
        'sds_dir': sds_subdir,
        'net': {
            'network': net,
            'station': stations,
            'component': 'Z'
        },
        'co': {
            'subdir': corr_subdir,
            'allow_different_params': True,
            'read_start': r_start_str,
            'read_end': r_end_str,
            'read_len': r_len,
            'read_inc': r_inc,
            'sampling_rate': fs,
            'remove_response': False,
            'joint_norm': False,
            'combination_method': comb_method,
            'preProcessing': [],
            'preprocess_subdiv': False,
            'subdivision': {
                'corr_inc': int(corr_inc),
                'corr_len': int(corr_len),
                'recombine_subdivision': False
            },
            'corr_args': {
                'TDpreProcessing': td_list,
                'FDpreProcessing': fd_list,
                'lengthToSave': float(max_lag),
                'center_correlation': True,
                'normalize_correlation': True,
                'combinations': []
            }
        }
    }
    return params


def build_monitor_config(pref_manager, proj_dir, corr_subdir, dv_subdir, log_subdir='log', start_date=None, end_date=None):
    ''' Build a SeisMIC monitor configuration dictionary from pSysmon preferences. '''
    tw_start = float(pref_manager.get_value('tw_start') or 0.05)
    tw_len = float(pref_manager.get_value('tw_len') or 0.20)
    sides = pref_manager.get_value('sides') or 'both'
    dv_method = pref_manager.get_value('dv_method') or 'stretch_multi'
    ref_type = pref_manager.get_value('ref_type') or 'mean'
    dv_max = float(pref_manager.get_value('dv_max') or 0.05)
    dv_step = float(pref_manager.get_value('dv_step') or 0.0005)
    min_coherence = float(pref_manager.get_value('min_coherence') or 0.5)
    freq_min = float(pref_manager.get_value('freq_min') or 2.0)
    freq_max = float(pref_manager.get_value('freq_max') or 25.0)

    # Format dates
    s_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S.0") if start_date else "2023-05-01 00:00:00.0"
    e_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S.0") if end_date else "2023-05-03 00:00:00.0"

    stretch_steps = int(round(2.0 * dv_max / dv_step)) + 1 if dv_step > 0 else 101

    params = {
        'proj_dir': proj_dir,
        'log_subdir': log_subdir,
        'log_level': 'INFO',
        'fig_subdir': 'figures',
        'net': {
            'network': '*',
            'station': '*',
            'component': 'Z'
        },
        'co': {
            'subdir': corr_subdir,
        },
        'dv': {
            'subdir': dv_subdir,
            'plot_vel_change': False,
            'start_date': s_date_str,
            'end_date': e_date_str,
            'win_len': 86400,
            'date_inc': 86400,
            'freq_min': freq_min,
            'freq_max': freq_max,
            'tw_start': tw_start,
            'tw_len': tw_len,
            'sides': sides,
            'compute_tt': False,
            'stretch_range': dv_max,
            'stretch_steps': stretch_steps,
            'min_coherence': min_coherence,
            'return_sim_mat': True,
            'dt_ref': {
                'win_inc': 0,
                'method': ref_type,
                'percentile': 50
            },
            'preprocessing': []
        }
    }
    return params
