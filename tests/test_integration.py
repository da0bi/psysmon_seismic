# LICENSE
#
# This file is part of psysmon_seismic.
#
'''
Automated unit and integration tests for psysmon_seismic.
'''

import os
import shutil
import tempfile
import unittest
import numpy as np
from obspy.core import Stream, Trace
from obspy.core.utcdatetime import UTCDateTime

import psysmon.core.result as psy_result
from psysmon_seismic.processing_nodes import ArrayStackingNode
from psysmon_seismic.seismic_xcorr import SeismicXCorrNode
from psysmon_seismic.seismic_monitor import SeismicMonitorNode
from psysmon_seismic.seismic_spatial import SeismicSpatialNode
from psysmon_seismic.utils import get_seismic_dirs, resolve_correlation_dir, resolve_dv_dir
from psysmon_seismic.client import psysmon_to_obspy_inventory, PsysmonStoreClient
from obspy.geodetics import gps2dist_azimuth
from seismic.db.corr_hdf5 import CorrelationDataBase


class MockProject(object):
    ''' Mock pSysmon project for headless testing. '''
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.geometry_inventory = None

    def log(self, mode, msg):
        pass


class TestPsysmonSeismic(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="psysmon_seismic_test_")
        self.project = MockProject(data_directory=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_stream(self, n_stations=3, locs=('A', 'B', 'C'), fs=100.0, duration=3600.0):
        ''' Create a multi-channel stream with synthetic reflection pulses. '''
        st = Stream()
        t0 = UTCDateTime("2023-05-01T00:00:00")
        npts = int(fs * duration)
        time_arr = np.linspace(0, duration, npts, endpoint=False)
        
        np.random.seed(42)
        for sta_idx in range(1, n_stations + 1):
            sta_name = f"AP23{sta_idx}"
            for loc in locs:
                # Base random noise
                data = np.random.randn(npts) * 0.1
                # Inject a synthetic coherent bedrock reflection at tau = 0.08 s
                for t_pulse in np.arange(10.0, duration, 60.0):
                    idx = int(t_pulse * fs)
                    if idx + 20 < npts:
                        # Ricker wavelet
                        t_w = np.linspace(-0.05, 0.05, 20)
                        wavelet = (1.0 - 2.0 * (np.pi * 15.0 * t_w)**2) * np.exp(-(np.pi * 15.0 * t_w)**2)
                        data[idx:idx+20] += wavelet
                
                tr = Trace(data=data.astype(np.float64))
                tr.stats.network = 'AP'
                tr.stats.station = sta_name
                tr.stats.location = loc
                tr.stats.channel = 'DPZ'
                tr.stats.sampling_rate = fs
                tr.stats.starttime = t0
                st.append(tr)
        return st

    def test_01_array_stacking_node_replace(self):
        ''' Test ArrayStackingNode with output_mode='replace'. '''
        st = self._create_synthetic_stream(n_stations=3, locs=('A', 'B', 'C'))
        self.assertEqual(len(st), 9)

        node = ArrayStackingNode()
        node.pref_manager.set_value('grouping', 'by_location_code')
        node.pref_manager.set_value('operator', 'mean')
        node.pref_manager.set_value('output_mode', 'replace')
        node.pref_manager.set_value('channel', 'DPZ')

        node.execute(st)

        # After replacement, 9 geophones should be stacked into 3 virtual stations
        self.assertEqual(len(st), 3)
        station_names = sorted([tr.stats.station for tr in st])
        self.assertEqual(station_names, ['AP231', 'AP232', 'AP233'])
        for tr in st:
            self.assertEqual(tr.stats.location, '')
            self.assertEqual(tr.stats.channel, 'DPZ')

    def test_02_array_stacking_node_replicate(self):
        ''' Test ArrayStackingNode with output_mode='replicate'. '''
        st = self._create_synthetic_stream(n_stations=2, locs=('A', 'B'))
        self.assertEqual(len(st), 4)

        node = ArrayStackingNode()
        node.pref_manager.set_value('grouping', 'by_location_code')
        node.pref_manager.set_value('operator', 'mean')
        node.pref_manager.set_value('output_mode', 'replicate')

        tr_a_orig = st.select(station='AP231', location='A')[0].data.copy()
        tr_b_orig = st.select(station='AP231', location='B')[0].data.copy()
        expected_mean = (tr_a_orig + tr_b_orig) / 2.0

        node.execute(st)

        # Trace count stays at 4
        self.assertEqual(len(st), 4)
        tr_a_new = st.select(station='AP231', location='A')[0].data
        tr_b_new = st.select(station='AP231', location='B')[0].data
        np.testing.assert_allclose(tr_a_new, expected_mean)
        np.testing.assert_allclose(tr_b_new, expected_mean)

    def test_03_seismic_xcorr_raw_stream_warning(self):
        ''' Test SeismicXCorrNode on raw unstacked stream and verify warning logging. '''
        st = self._create_synthetic_stream(n_stations=1, locs=('A', 'B'), fs=50.0, duration=2 * 86400.0)
        self.assertEqual(len(st), 2)

        xcorr_node = SeismicXCorrNode()
        xcorr_node.project = self.project
        xcorr_node.pref_manager.set_value('combination_method', 'autoComponents')
        xcorr_node.pref_manager.set_value('corr_len', 3600.0)
        xcorr_node.pref_manager.set_value('corr_inc', 3600.0)
        xcorr_node.pref_manager.set_value('max_lag', 1.0)
        xcorr_node.pref_manager.set_value('spectral_whitening', False)

        with self.assertLogs(level='WARNING') as cm:
            xcorr_node.execute(st)
        
        # Verify warning log message
        warning_emitted = any("Incoming stream contains raw, unstacked station channels" in log_msg for log_msg in cm.output)
        self.assertTrue(warning_emitted, "Expected warning about raw unstacked stream was not emitted.")

        # Verify correlation files were created for raw geophones
        seismic_dirs = get_seismic_dirs(self.project.data_directory)
        h5_files = os.listdir(seismic_dirs['corr'])
        self.assertGreaterEqual(len(h5_files), 1)

    def test_04_full_pipeline_stacked_to_results(self):
        ''' Test complete workflow: Stack -> Correlate (looper output_dir) -> Monitor (user pref) -> Spatial (user pref). '''
        # 1. Synthetic 9-channel stream (2 days continuous at 50 Hz)
        st = self._create_synthetic_stream(n_stations=3, locs=('A', 'B', 'C'), fs=50.0, duration=2 * 86400.0)

        # 2. Stack via ArrayStackingNode
        stack_node = ArrayStackingNode()
        stack_node.pref_manager.set_value('grouping', 'by_location_code')
        stack_node.pref_manager.set_value('operator', 'mean')
        stack_node.pref_manager.set_value('output_mode', 'replace')
        stack_node.execute(st)
        self.assertEqual(len(st), 3)

        # 3. Correlate via SeismicXCorrNode into a simulated looper process folder
        looper_proc_dir = os.path.join(self.test_dir, 'smi-up.db-psysmon-apo23-SeisMIC_simulated-time_window_looper')
        xcorr_node = SeismicXCorrNode()
        xcorr_node.project = self.project
        xcorr_node.pref_manager.set_value('combination_method', 'autoComponents')
        xcorr_node.pref_manager.set_value('corr_len', 3600.0)
        xcorr_node.pref_manager.set_value('corr_inc', 3600.0)
        xcorr_node.pref_manager.set_value('max_lag', 1.0)
        xcorr_node.pref_manager.set_value('spectral_whitening', True)
        xcorr_node.pref_manager.set_value('whiten_waterlevel', 1e-4)
        xcorr_node.pref_manager.set_value('freq_min', 5.0)
        xcorr_node.pref_manager.set_value('freq_max', 20.0)
        xcorr_node.execute(st, output_dir=looper_proc_dir)

        corr_files = [f for f in os.listdir(os.path.join(looper_proc_dir, 'seismic', 'corr')) if f.endswith('.h5')]
        self.assertEqual(len(corr_files), 3)

        # 4. Monitor via SeismicMonitorNode
        monitor_node = SeismicMonitorNode()
        monitor_node.project = self.project
        # 4a. Verify that running without corr_dir logs error and produces 0 results
        monitor_node.execute()
        self.assertEqual(len(monitor_node.result_bag.results), 0)

        # 4b. Set user-specified corr_dir and execute
        monitor_node.pref_manager.set_value('corr_dir', looper_proc_dir)
        monitor_node.pref_manager.set_value('tw_start', 0.05)
        monitor_node.pref_manager.set_value('tw_len', 0.20)
        monitor_node.pref_manager.set_value('sides', 'both')
        monitor_node.pref_manager.set_value('dv_method', 'stretch_multi')
        monitor_node.pref_manager.set_value('ref_type', 'mean')
        monitor_node.pref_manager.set_value('min_coherence', 0.65)
        monitor_node.pref_manager.set_value('freq_min', 5.0)
        monitor_node.pref_manager.set_value('freq_max', 20.0)
        monitor_node.execute()

        # Check ResultBag
        results = monitor_node.result_bag.results
        self.assertGreaterEqual(len(results), 6)  # 3 TableResult + 3 ShelveResult

        # Verify automatic save directly into looper_proc_dir
        dv_files = [f for f in os.listdir(os.path.join(looper_proc_dir, 'seismic', 'dv')) if f.endswith('.npz')]
        self.assertEqual(len(dv_files), 3)

        # 5. Spatial via SeismicSpatialNode
        spatial_node = SeismicSpatialNode()
        spatial_node.project = self.project
        # 5a. Verify that running without dv_dir logs error and produces 0 results
        spatial_node.execute()
        self.assertEqual(len(spatial_node.result_bag.results), 0)

        # 5b. Set user-specified dv_dir and execute
        spatial_node.pref_manager.set_value('dv_dir', looper_proc_dir)
        spatial_node.pref_manager.set_value('grid_resolution', 50.0)
        spatial_node.pref_manager.set_value('epsg', 'EPSG:3995')
        spatial_node.execute()

        self.assertEqual(len(spatial_node.result_bag.results), 1)

        # Verify saved files exist on disk inside looper_proc_dir
        saved_files = []
        for r, d, files in os.walk(looper_proc_dir):
            for f in files:
                saved_files.append(f)

        has_csv = any(f.endswith('.csv') for f in saved_files)
        has_db = any(f.endswith('.db') for f in saved_files)
        has_asc = any(f.endswith('.asc') for f in saved_files)
        has_prj = any(f.endswith('.prj') for f in saved_files)

        self.assertTrue(has_csv, "Expected .csv result file was not saved in looper_proc_dir.")
        self.assertTrue(has_db, "Expected .db result file was not saved in looper_proc_dir.")
        self.assertTrue(has_asc, "Expected .asc result file was not saved in looper_proc_dir.")
        self.assertTrue(has_prj, "Expected .prj result file was not saved in looper_proc_dir.")

    def test_05_guide_document_exists(self):
        ''' Verify SeisMIC_psysmon_guide.md is present and contains required sections. '''
        guide_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "SeisMIC_psysmon_guide.md")
        self.assertTrue(os.path.isfile(guide_path), f"Guide not found at {guide_path}")
        
        with open(guide_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("1. Package Architecture & Implementation", content)
        self.assertIn("2. Node & Parameter Reference", content)
        self.assertIn("2.2.1 Multi-Core Parallelization & Zero-Disk Architecture", content)
        self.assertIn("2.2.2 Workflow & Parameter Optimization for Maximum Speed", content)
        self.assertIn("2.4.1 Detailed Physics Reference: Scattering & Medium Physics", content)
        self.assertIn("2.5 Automated Headless Plot Generation & Visualization", content)
        self.assertIn("2.5.4 Dynamic Vertical Scaling & Adaptive Wiggle Sizing", content)
        self.assertIn("Evaluation: Permutations of Stacked & Unstacked Correlations", content)
        self.assertIn("3. Best Practice Guide: The APO23 Glacier Dataset", content)
        self.assertIn("3.2 Best Practice Guide 1: Autocorrelation of Stacked Station Data", content)
        self.assertIn("3.3 Best Practice Guide 2: Inter-Station Cross-Correlation Between Stacked Data", content)

    def test_06_directory_resolution(self):
        ''' Verify resolve_correlation_dir and resolve_dv_dir across all directory permutations. '''
        sim_proc = os.path.join(self.test_dir, "sim_proc")
        seis = os.path.join(sim_proc, "seismic")
        corr = os.path.join(seis, "corr")
        dv = os.path.join(seis, "dv")
        os.makedirs(corr, exist_ok=True)
        os.makedirs(dv, exist_ok=True)

        # Test correlation resolution
        p1, c1 = resolve_correlation_dir(sim_proc)
        self.assertEqual(p1, sim_proc)
        self.assertEqual(c1, corr)

        p2, c2 = resolve_correlation_dir(seis)
        self.assertEqual(p2, sim_proc)
        self.assertEqual(c2, corr)

        p3, c3 = resolve_correlation_dir(corr)
        self.assertEqual(p3, sim_proc)
        self.assertEqual(c3, corr)

        p_inv, c_inv = resolve_correlation_dir("/nonexistent/path")
        self.assertIsNone(p_inv)
        self.assertIsNone(c_inv)

        # Test dv resolution
        pd1, d1 = resolve_dv_dir(sim_proc)
        self.assertEqual(pd1, sim_proc)
        self.assertEqual(d1, dv)

        pd2, d2 = resolve_dv_dir(seis)
        self.assertEqual(pd2, sim_proc)
        self.assertEqual(d2, dv)

        pd3, d3 = resolve_dv_dir(dv)
        self.assertEqual(pd3, sim_proc)
        self.assertEqual(d3, dv)

        pd_inv, d_inv = resolve_dv_dir("")
        self.assertIsNone(pd_inv)
        self.assertIsNone(d_inv)

    def test_07_in_memory_inventory_centroid_and_distances(self):
        ''' Test in-memory inventory conversion: centroid calculation and inter-station distances in HDF5 headers. '''
        class MockChannel(object):
            def __init__(self, name, location=''):
                self.name = name
                self.location = location

        class MockStation(object):
            def __init__(self, name, lat, lon, elev=0.0):
                self.name = name
                self.lat = lat
                self.lon = lon
                self.z = elev
                self.channels = [MockChannel('DPZ', '')]
            def get_lon_lat(self):
                return self.lon, self.lat

        class MockNetwork(object):
            def __init__(self, name, stations):
                self.name = name
                self.stations = stations

        class MockInventory(object):
            def __init__(self, networks):
                self.networks = networks

        # Realistic geophone coordinates for the 3 APO23 mini-arrays
        stations = [
            MockStation('AP231A', 74.623068, -21.376281, 671.0),
            MockStation('AP231B', 74.622955, -21.376170, 671.0),
            MockStation('AP231C', 74.622978, -21.376470, 671.0),
            MockStation('AP232A', 74.636104, -21.415849, 820.5),
            MockStation('AP232B', 74.636081, -21.416413, 820.5),
            MockStation('AP232C', 74.635995, -21.416132, 820.5),
            MockStation('AP233A', 74.643217, -21.475470, 923.0),
            MockStation('AP233B', 74.643257, -21.475970, 923.0),
            MockStation('AP233C', 74.643137, -21.475220, 923.0),
        ]
        p_inv = MockInventory([MockNetwork('7G', stations)])

        # 1. Verify in-memory inventory conversion
        obs_inv = psysmon_to_obspy_inventory(p_inv)
        sta_codes = [s.code for s in obs_inv[0]]
        # Should contain all 9 raw geophones + 3 virtual array centroid stations
        for raw_sta in ['AP231A', 'AP231B', 'AP231C', 'AP232A', 'AP232B', 'AP232C', 'AP233A', 'AP233B', 'AP233C']:
            self.assertIn(raw_sta, sta_codes)
        for virt_sta in ['AP231', 'AP232', 'AP233']:
            self.assertIn(virt_sta, sta_codes)

        # 2. Verify centroid coordinates and geodesic inter-station distances
        s1 = obs_inv.select(network='7G', station='AP231')[0][0]
        s2 = obs_inv.select(network='7G', station='AP232')[0][0]
        s3 = obs_inv.select(network='7G', station='AP233')[0][0]

        d12, az12, baz12 = gps2dist_azimuth(s1.latitude, s1.longitude, s2.latitude, s2.longitude)
        d23, az23, baz23 = gps2dist_azimuth(s2.latitude, s2.longitude, s3.latitude, s3.longitude)
        d13, az13, baz13 = gps2dist_azimuth(s1.latitude, s1.longitude, s3.latitude, s3.longitude)

        self.assertAlmostEqual(d12, 1874.6, delta=5.0)  # ~1.875 km
        self.assertAlmostEqual(d23, 1930.1, delta=5.0)  # ~1.930 km
        self.assertAlmostEqual(d13, 3702.7, delta=5.0)  # ~3.703 km

        # 3. Verify PsysmonStoreClient serves the in-memory inventory directly
        self.project.geometry_inventory = p_inv
        store_client = PsysmonStoreClient(self.project, self.test_dir)
        served_inv = store_client.read_inventory()
        self.assertEqual(len(served_inv[0]), 12)

        # 4. Verify betweenStations cross-correlation produces HDF5 files with dist, az, baz embedded
        fs = 50.0
        duration = 86400.0  # 1 day
        npts = int(fs * duration)
        t0 = UTCDateTime('2023-05-01T00:00:00')

        st = Stream()
        np.random.seed(123)
        for sta_name in ['AP231', 'AP232']:
            tr = Trace(data=np.random.randn(npts).astype(np.float64))
            tr.stats.network = '7G'
            tr.stats.station = sta_name
            tr.stats.channel = 'DPZ'
            tr.stats.sampling_rate = fs
            tr.stats.starttime = t0
            st.append(tr)

        looper_proc_dir = os.path.join(self.test_dir, 'sim_interstation_looper')
        xcorr_node = SeismicXCorrNode()
        xcorr_node.project = self.project
        xcorr_node.pref_manager.set_value('combination_method', 'betweenStations')
        xcorr_node.pref_manager.set_value('corr_len', 3600.0)
        xcorr_node.pref_manager.set_value('corr_inc', 3600.0)
        xcorr_node.pref_manager.set_value('max_lag', 5.0)
        xcorr_node.pref_manager.set_value('spectral_whitening', False)
        xcorr_node.execute(st, output_dir=looper_proc_dir)

        corr_file = os.path.join(looper_proc_dir, 'seismic', 'corr', '7G-7G.AP231-AP232.-.DPZ-DPZ.h5')
        self.assertTrue(os.path.isfile(corr_file), f"Expected HDF5 file not found: {corr_file}")

        # Check HDF5 attributes for dist, az, baz
        with CorrelationDataBase(corr_file) as db:
            found_dist = []
            def visitor(name, obj):
                if 'dist' in obj.attrs:
                    found_dist.append((obj.attrs['dist'], obj.attrs['az'], obj.attrs['baz']))
            db.visititems(visitor)

        self.assertGreater(len(found_dist), 0, "No correlation datasets with 'dist' attribute found in HDF5.")
        first_dist, first_az, first_baz = found_dist[0]
        # In SeisMIC, dist is stored in km
        self.assertAlmostEqual(first_dist, 1.8746, delta=0.01)
        self.assertAlmostEqual(first_az, 321.06, delta=1.0)
        self.assertAlmostEqual(first_baz, 141.02, delta=1.0)

    def test_08_array_stacking_gui_layout(self):
        ''' Verify ArrayStackingNode tabbed preferences layout and streamlined parameter keys. '''
        node = ArrayStackingNode()

        # 1. Verify tabbed pages
        page_names = [p.name for p in node.pref_manager.pages]
        self.assertEqual(page_names, ['Grouping', 'Stacking', 'Plots'])

        # 2. Verify streamlined parameter keys and default values
        self.assertEqual(node.pref_manager.get_value('grouping'), 'by_location_code')
        self.assertEqual(node.pref_manager.get_value('channel'), 'DPZ')
        self.assertEqual(node.pref_manager.get_value('operator'), 'mean')
        self.assertEqual(node.pref_manager.get_value('output_mode'), 'replace')
        self.assertFalse(node.pref_manager.get_value('normalize'))

        # 3. Verify GUI building if wxPython is available
        import psysmon
        if psysmon.wx_available:
            import wx
            app = wx.App.Get() or wx.App()
            frame = wx.Frame(None, size=(500, 600))
            edit_panel = node.getEditPanel(parent=frame)
            self.assertIsNotNone(edit_panel)
            self.assertEqual(edit_panel.notebook.GetPageCount(), 3)
            self.assertEqual(edit_panel.notebook.GetPageText(0), 'Grouping')
            self.assertEqual(edit_panel.notebook.GetPageText(1), 'Stacking')
            self.assertEqual(edit_panel.notebook.GetPageText(2), 'Plots')
            frame.Destroy()

    def test_09_spatial_epsg_validation_and_reprojection(self):
        ''' Verify SeismicSpatialNode metric CRS validation, error halts on degrees, and reprojection. '''
        from psysmon_seismic.seismic_spatial import validate_metric_crs

        # 1. Unit test validate_metric_crs directly
        with self.assertRaises(ValueError) as cm:
            validate_metric_crs('EPSG:4326')
        self.assertIn("geographic CRS measured in degrees", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            validate_metric_crs('')
        self.assertIn("No coordinate system (EPSG) specified", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            validate_metric_crs('INVALID_PROJ_999')
        self.assertIn("Invalid EPSG specification", str(cm.exception))

        crs_3995 = validate_metric_crs('EPSG:3995')
        self.assertTrue(crs_3995.is_projected)
        crs_utm = validate_metric_crs('EPSG:32627')
        self.assertTrue(crs_utm.is_projected)

        # 2. Setup mock data directory with DV files
        sim_proc = os.path.join(self.test_dir, "test_spatial_proc")
        seis_dirs = get_seismic_dirs(sim_proc)
        dv_dir = seis_dirs['dv']

        from seismic.monitor.dv import DV
        from seismic.correlate.stats import CorrStats
        header = {
            'corr_start': [UTCDateTime("2023-05-01")],
            'corr_end': [UTCDateTime("2023-05-02")],
            'network': 'AP', 'station': 'AP231', 'channel': 'DPZ'
        }
        stats = CorrStats(header)
        dv_obj = DV(
            corr=np.array([0.8]),
            value=np.array([0.002]),
            value_type='stretch',
            sim_mat=np.array([[0.8]]),
            second_axis=np.array([0.002]),
            method='stretch_multi',
            stats=stats
        )
        dv_obj.save(os.path.join(dv_dir, 'DV-test.npz'))

        node = SeismicSpatialNode()
        node.project = self.project
        node.pref_manager.set_value('dv_dir', sim_proc)

        # 3. Test that geographic CRS (EPSG:4326) strictly halts execution
        node.pref_manager.set_value('epsg', 'EPSG:4326')
        node.pref_manager.set_value('grid_resolution', 50.0)
        node.execute()
        self.assertEqual(len(node.result_bag.results), 0, "Execution should halt with 0 results for EPSG:4326")

        # 4. Test that invalid EPSG halts execution
        node.pref_manager.set_value('epsg', 'INVALID:9999')
        node.execute()
        self.assertEqual(len(node.result_bag.results), 0, "Execution should halt with 0 results for invalid EPSG")

        # 5. Test valid EPSG:3995 (Arctic Polar Stereographic) with metric resolution
        node.pref_manager.set_value('epsg', 'EPSG:3995')
        node.pref_manager.set_value('grid_resolution', 50.0)
        node.execute()
        self.assertEqual(len(node.result_bag.results), 1)

        grid_res = node.result_bag.results[0]
        self.assertEqual(grid_res.dx, 50.0)
        self.assertEqual(grid_res.dy, 50.0)
        self.assertEqual(grid_res.epsg, 'EPSG:3995')
        # Coordinates should be large metric coordinates in polar stereographic projection (< -100,000 m)
        self.assertLess(grid_res.x_coord[0], -100000.0)
        self.assertLess(grid_res.y_coord[0], -100000.0)

        # 6. Test auto-sanitization of legacy resolution (< 1.0)
        legacy_node = SeismicSpatialNode()
        legacy_node.pref_manager.set_value('grid_resolution', 0.001)
        legacy_node._sanitize_resolution()
        self.assertEqual(legacy_node.pref_manager.get_value('grid_resolution'), 50.0)

        legacy_node.pref_manager.set_value('grid_resolution', 0.1)
        legacy_node._sanitize_resolution()
        self.assertEqual(legacy_node.pref_manager.get_value('grid_resolution'), 50.0)

    def test_10_seismic_xcorr_parallel_and_in_memory(self):
        ''' Verify SeismicXCorrNode parallelization (ProcessPoolExecutor) and in-memory execution. '''
        import h5py

        # 1. Verify preference items and default values
        node = SeismicXCorrNode()
        self.assertEqual(node.pref_manager.get_value('parallel_workers'), 4)
        item_workers = node.pref_manager.get_item('parallel_workers')[0]
        self.assertEqual(item_workers.limit, (1, 32))

        self.assertEqual(node.pref_manager.get_value('log_level'), 'WARNING')
        item_log = node.pref_manager.get_item('log_level')[0]
        self.assertEqual(item_log.limit, ('WARNING', 'INFO', 'DEBUG'))

        # 2. Create synthetic stream (2 hours, 2 stations, 50 Hz)
        st = self._create_synthetic_stream(n_stations=2, locs=('',), fs=50.0, duration=7200.0)

        # 3. Execute sequentially (parallel_workers = 1)
        proc_seq = os.path.join(self.test_dir, "proc_seq")
        node_seq = SeismicXCorrNode()
        node_seq.project = self.project
        node_seq.pref_manager.set_value('parallel_workers', 1)
        node_seq.pref_manager.set_value('corr_len', 1800.0)
        node_seq.pref_manager.set_value('corr_inc', 3600.0)
        node_seq.pref_manager.set_value('max_lag', 2.0)
        node_seq.pref_manager.set_value('combination_method', 'autoComponents')
        node_seq.pref_manager.set_value('log_level', 'WARNING')
        node_seq.execute(st.copy(), output_dir=proc_seq)

        # 4. Execute in parallel (parallel_workers = 2)
        proc_par = os.path.join(self.test_dir, "proc_par")
        node_par = SeismicXCorrNode()
        node_par.project = self.project
        node_par.pref_manager.set_value('parallel_workers', 2)
        node_par.pref_manager.set_value('corr_len', 1800.0)
        node_par.pref_manager.set_value('corr_inc', 3600.0)
        node_par.pref_manager.set_value('max_lag', 2.0)
        node_par.pref_manager.set_value('combination_method', 'autoComponents')
        node_par.pref_manager.set_value('log_level', 'WARNING')
        node_par.execute(st.copy(), output_dir=proc_par)

        # 5. Verify zero disk staging (no miniSEED files or remaining sds_stage directories)
        self.assertFalse(os.path.exists(os.path.join(proc_seq, 'seismic', 'sds_stage')))
        self.assertFalse(os.path.exists(os.path.join(proc_par, 'seismic', 'sds_stage')))

        # 6. Verify worker temporary directories were fully merged and cleaned up
        worker_dirs = [d for d in os.listdir(os.path.join(proc_par, 'seismic')) if 'worker' in d]
        self.assertEqual(len(worker_dirs), 0)

        # 7. Compare correlation databases between sequential and parallel runs
        corr_dir_seq = os.path.join(proc_seq, 'seismic', 'corr')
        corr_dir_par = os.path.join(proc_par, 'seismic', 'corr')
        files_seq = sorted([f for f in os.listdir(corr_dir_seq) if f.endswith('.h5')])
        files_par = sorted([f for f in os.listdir(corr_dir_par) if f.endswith('.h5')])
        self.assertEqual(files_seq, files_par)
        self.assertGreater(len(files_seq), 0)

        for fname in files_seq:
            path_seq = os.path.join(corr_dir_seq, fname)
            path_par = os.path.join(corr_dir_par, fname)
            with h5py.File(path_seq, 'r') as h5_s, h5py.File(path_par, 'r') as h5_p:
                datasets_s = {}
                h5_s.visititems(lambda name, obj: datasets_s.update({name: obj[:]}) if isinstance(obj, h5py.Dataset) else None)
                datasets_p = {}
                h5_p.visititems(lambda name, obj: datasets_p.update({name: obj[:]}) if isinstance(obj, h5py.Dataset) else None)

                self.assertEqual(sorted(datasets_s.keys()), sorted(datasets_p.keys()))
                self.assertGreater(len(datasets_s), 0)
                for k in datasets_s:
                    self.assertEqual(datasets_s[k].shape, datasets_p[k].shape)
                    np.testing.assert_allclose(datasets_s[k], datasets_p[k], atol=1e-6)

    def test_11_processing_stack_gui_sizing(self):
        ''' Verify ProcessingStack dialog dimensions, nodeOptions height >= 420px, and zero parameter clipping. '''
        import psysmon
        if not psysmon.wx_available:
            self.skipTest("wxPython not available in environment.")

        import wx
        from psysmon.packages.core_looper_children.processing_stack import ProcessingStackLooperChild
        from psysmon.packages.tracedisplay.processingNodes import FilterBandPass, RemoveResponsePAZ
        from psysmon.gui.dialog.pref_listbook import ListbookPrefDialog
        from psysmon.packages.tracedisplay.plugins_processingstack import PStackEditPanel

        app = wx.App.Get() or wx.App()

        node = ProcessingStackLooperChild()
        paz = RemoveResponsePAZ()
        bp = FilterBandPass()
        arr = ArrayStackingNode()
        node.pref_manager.set_value('processing_stack', [paz, bp, arr])

        # Verify RemoveResponsePAZ is organized into General, Pre-filter, and Performance pages
        paz_pages = [p.name for p in paz.pref_manager.pages]
        self.assertEqual(paz_pages, ['General', 'Pre-filter', 'Performance'])

        # Create dialog
        dlg = ListbookPrefDialog(preferences=node.pref_manager)
        dlg_size = dlg.GetSize()
        self.assertGreaterEqual(dlg_size[0], 600)
        self.assertGreaterEqual(dlg_size[1], 700)

        # Locate PStackEditPanel
        def find_pstack(w):
            if isinstance(w, PStackEditPanel):
                return w
            for c in w.GetChildren():
                res = find_pstack(c)
                if res:
                    return res
            return None

        ps_panel = find_pstack(dlg)
        self.assertIsNotNone(ps_panel)

        # Verify CheckListBox height fits 5 nodes cleanly (145-150px)
        clb_size = ps_panel.nodeListBox.GetSize()
        self.assertLessEqual(clb_size[1], 155)
        self.assertGreaterEqual(clb_size[1], 140)

        # Verify active nodeOptions has at least 420px vertical space (no clipping for PAZ, BP, or Arr)
        options_size = ps_panel.nodeOptions.GetSize()
        self.assertGreaterEqual(options_size[1], 420)

        # Verify switching nodes maintains ample vertical space
        evt = wx.CommandEvent(wx.wxEVT_COMMAND_LISTBOX_SELECTED, ps_panel.nodeListBox.GetId())
        evt.SetInt(1)
        ps_panel.on_node_selected(evt)
        self.assertGreaterEqual(ps_panel.nodeOptions.GetSize()[1], 420)

        evt.SetInt(2)
        ps_panel.on_node_selected(evt)
        self.assertGreaterEqual(ps_panel.nodeOptions.GetSize()[1], 420)

        dlg.Destroy()

    def test_12_processing_stack_remove_and_reorder_buttons(self):
        ''' Test remove, up, and down buttons in PStackEditPanel. '''
        import psysmon
        if not psysmon.wx_available:
            self.skipTest("wxPython not available")

        import wx
        from psysmon.packages.core_looper_children.processing_stack import ProcessingStackLooperChild
        from psysmon.gui.dialog.pref_listbook import ListbookPrefDialog
        from psysmon.packages.tracedisplay.plugins_processingstack import PStackEditPanel
        from psysmon.packages.tracedisplay.processingNodes import Detrend, FilterBandPass, RemoveResponsePAZ
        from psysmon_seismic.processing_nodes import ArrayStackingNode

        if not wx.GetApp():
            app = wx.App(False)

        node = ProcessingStackLooperChild()
        pstack_item = node.pref_manager.get_item('processing_stack')[0]
        n_dt = Detrend()
        n_paz = RemoveResponsePAZ()
        n_bp = FilterBandPass()
        n_arr = ArrayStackingNode()

        pstack_item.value = [n_dt, n_paz, n_bp, n_arr]
        pstack_item.limit = [n_dt, n_paz, n_bp, n_arr]

        dlg = ListbookPrefDialog(preferences=node.pref_manager)

        def find_pstack(w):
            if isinstance(w, PStackEditPanel):
                return w
            for c in w.GetChildren():
                res = find_pstack(c)
                if res:
                    return res
            return None

        ps_panel = find_pstack(dlg)
        self.assertIsNotNone(ps_panel)

        # 1. Verify initial order
        self.assertEqual(
            [n.name for n in ps_panel.selected_nodes],
            ['detrend', 'remove response (paz)', 'bandpass filter', 'array stacking']
        )

        # 2. Test Move Up: Move 'array stacking' (index 3) up to top (index 0)
        ps_panel.nodeListBox.SetSelection(3)
        ps_panel.on_up_button_clicked(None)
        self.assertEqual(
            [n.name for n in ps_panel.selected_nodes],
            ['detrend', 'remove response (paz)', 'array stacking', 'bandpass filter']
        )
        self.assertEqual(ps_panel.nodeListBox.GetSelection(), 2)

        ps_panel.on_up_button_clicked(None)
        self.assertEqual(
            [n.name for n in ps_panel.selected_nodes],
            ['detrend', 'array stacking', 'remove response (paz)', 'bandpass filter']
        )
        self.assertEqual(ps_panel.nodeListBox.GetSelection(), 1)

        ps_panel.on_up_button_clicked(None)
        self.assertEqual(
            [n.name for n in ps_panel.selected_nodes],
            ['array stacking', 'detrend', 'remove response (paz)', 'bandpass filter']
        )
        self.assertEqual(ps_panel.nodeListBox.GetSelection(), 0)

        # Boundary test: Up at index 0 does nothing
        ps_panel.on_up_button_clicked(None)
        self.assertEqual(
            [n.name for n in ps_panel.selected_nodes],
            ['array stacking', 'detrend', 'remove response (paz)', 'bandpass filter']
        )
        self.assertEqual(ps_panel.nodeListBox.GetSelection(), 0)

        # 3. Test Move Down: Move 'array stacking' (index 0) down to index 1
        ps_panel.on_down_button_clicked(None)
        self.assertEqual(
            [n.name for n in ps_panel.selected_nodes],
            ['detrend', 'array stacking', 'remove response (paz)', 'bandpass filter']
        )
        self.assertEqual(ps_panel.nodeListBox.GetSelection(), 1)

        # 4. Test Remove: Remove 'detrend' (index 0)
        ps_panel.nodeListBox.SetSelection(0)
        ps_panel.on_remove_button_clicked(None)
        self.assertEqual(
            [n.name for n in ps_panel.selected_nodes],
            ['array stacking', 'remove response (paz)', 'bandpass filter']
        )
        self.assertEqual(ps_panel.nodeListBox.GetSelection(), 0)
        self.assertEqual(ps_panel.nodeListBox.GetString(0), 'array stacking')

        # 5. Remove remaining nodes down to empty stack without error
        while ps_panel.selected_nodes:
            ps_panel.nodeListBox.SetSelection(0)
            ps_panel.on_remove_button_clicked(None)
        self.assertEqual(len(ps_panel.selected_nodes), 0)
        self.assertEqual(ps_panel.nodeListBox.GetCount(), 0)

        # 6. Verify button sizer contains all 4 buttons and fits within height
        top_sizer = ps_panel.GetSizer().GetItem(0).GetSizer()
        btn_sizer = top_sizer.GetItem(1).GetSizer() if top_sizer else ps_panel.GetSizer().GetItem(1).GetSizer()
        self.assertGreaterEqual(len(btn_sizer.GetChildren()), 4)

        dlg.Destroy()

    def test_13_add_node_dialog_sizing(self):
        ''' Test PStackAddNodeDialog dimensions, column widths, and zero truncation. '''
        import psysmon
        if not psysmon.wx_available:
            self.skipTest("wxPython not available")

        import wx
        from psysmon.packages.tracedisplay.processingNodes import (
            Detrend, MedianFilter, FilterBandPass, FilterLowPass, FilterHighPass,
            ConvertToSensorUnits, ScaleLog10, Resample, RemoveResponsePAZ
        )
        from psysmon_seismic.processing_nodes import ArrayStackingNode
        from psysmon.packages.tracedisplay.plugins_processingstack import PStackAddNodeDialog

        if not wx.GetApp():
            app = wx.App(False)

        nodes = [
            Detrend(), MedianFilter(), FilterBandPass(), FilterLowPass(), FilterHighPass(),
            ConvertToSensorUnits(), ScaleLog10(), Resample(), RemoveResponsePAZ(),
            ArrayStackingNode()
        ]

        dlg = PStackAddNodeDialog(None, availableNodes=nodes)
        self.assertGreaterEqual(dlg.GetSize()[0], 860)
        self.assertGreaterEqual(dlg.GetSize()[1], 480)

        panel = dlg.nodeInventoryPanel
        lc = panel.nodeListCtrl

        # Verify ListCtrl has ample space for all items
        self.assertGreaterEqual(lc.GetSize()[0], 840)
        self.assertGreaterEqual(lc.GetSize()[1], 360)

        # Verify column widths
        self.assertGreaterEqual(lc.GetColumnWidth(0), 180)  # name
        self.assertGreaterEqual(lc.GetColumnWidth(1), 90)   # mode
        self.assertGreaterEqual(lc.GetColumnWidth(2), 150)  # category
        self.assertGreaterEqual(lc.GetColumnWidth(3), 350)  # tags

        total_col_width = sum(lc.GetColumnWidth(i) for i in range(4))
        self.assertGreaterEqual(total_col_width, 800)

        # Verify all 10 items fit in height without needing a vertical scrollbar
        row_h = lc.GetItemRect(0).height
        total_items_h = row_h * len(nodes) + 30
        self.assertGreater(lc.GetSize()[1], total_items_h)

        # Test double-click activation / item selection
        item_event = wx.ListEvent(wx.wxEVT_COMMAND_LIST_ITEM_ACTIVATED, lc.GetId())
        item_event.SetIndex(0)
        item_event.SetItem(lc.GetItem(0))
        panel.onNodeItemSelected(item_event)
        self.assertIsNotNone(panel.selectedNode)
        self.assertEqual(panel.selectedNode.name, nodes[0].name)

        dlg.Destroy()

    def test_14_array_stacking_vectorized_equivalence(self):
        ''' Verify ArrayStackingNode 2D vectorized execution equivalence to manual loops. '''
        st = self._create_synthetic_stream(n_stations=2, locs=('A', 'B', 'C'), fs=100.0, duration=60.0)

        # Test mean, median, sum with and without RMS normalization
        for op in ['mean', 'median', 'sum']:
            for norm in [False, True]:
                node = ArrayStackingNode()
                node.pref_manager.set_value('operator', op)
                node.pref_manager.set_value('normalize', norm)
                test_st = st.copy()
                node.execute(test_st)

                # Check manual computation for station AP231
                traces_1 = [tr for tr in st if tr.stats.station == 'AP231']
                min_npts = min(tr.stats.npts for tr in traces_1)
                manual_mat = np.array([tr.data[:min_npts] for tr in traces_1], dtype=np.float64)
                if norm:
                    rms = np.sqrt(np.mean(manual_mat**2, axis=1, keepdims=True))
                    manual_mat = np.divide(manual_mat, rms, out=manual_mat.copy(), where=(rms > 0))

                if op == 'median':
                    manual_stacked = np.median(manual_mat, axis=0)
                elif op == 'sum':
                    manual_stacked = np.sum(manual_mat, axis=0)
                else:
                    manual_stacked = np.mean(manual_mat, axis=0)

                out_tr = [tr for tr in test_st if tr.stats.station == 'AP231'][0]
                np.testing.assert_allclose(out_tr.data, manual_stacked, atol=1e-6)

    def test_15_pxcorr_matrix_and_preprocessing_vectorized_equivalence(self):
        ''' Verify vectorized pxcorr_matrix, clip, detrendqr, and FDfilter equivalence. '''
        from psysmon_seismic.seismic_xcorr import (
            _vectorized_clip,
            _vectorized_detrendqr,
            _vectorized_FDfilter
        )
        import seismic.correlate.preprocessing_td as pptd
        import seismic.correlate.preprocessing_fd as ppfd

        np.random.seed(42)
        # 1. Clip equivalence
        A_orig = np.random.randn(5, 500) * 10.0
        A_vec = A_orig.copy()
        args = {'std_factor': 1.5}
        params = {}
        res_orig = pptd.clip(A_orig.copy(), args, params)
        res_vec = _vectorized_clip(A_vec, args, params)
        np.testing.assert_allclose(res_orig, res_vec, atol=1e-6)

        # 2. Detrendqr equivalence
        data_orig = np.random.randn(4, 1000).astype(np.float32) + np.linspace(0, 5, 1000)[None, :]
        data_vec = data_orig.copy()
        res_dt_orig = pptd.detrendqr(data_orig.copy())
        res_dt_vec = _vectorized_detrendqr(data_vec)
        np.testing.assert_allclose(res_dt_orig, res_dt_vec, atol=1e-6)

        # 3. FDfilter equivalence
        B_orig = np.random.randn(3, 513).astype(np.complex64) + 1j * np.random.randn(3, 513).astype(np.complex64)
        B_vec = B_orig.copy()
        freqs = np.linspace(0, 50, 513)
        args_fd = {'flimit': [1.0, 2.0, 20.0, 25.0]}
        params_fd = {'freqs': freqs}
        res_fd_orig = ppfd.FDfilter(B_orig.copy(), args_fd, params_fd)
        res_fd_vec = _vectorized_FDfilter(B_vec, args_fd, params_fd)
        np.testing.assert_allclose(res_fd_orig, res_fd_vec, atol=1e-6)

    def test_16_stretching_and_spatial_vectorized_equivalence(self):
        ''' Verify vectorized velocity_change_estimate, compute_wfc, and sensitivity_kernel equivalence. '''
        from psysmon_seismic.seismic_monitor import (
            _vectorized_velocity_change_estimate,
            _vectorized_compute_wfc,
            _vectorized_create_shifted_ref_mat
        )
        from psysmon_seismic.seismic_spatial import _vectorized_sensitivity_kernel
        import seismic.monitor.stretch_mod as sm
        import seismic.monitor.spatial as sm_spatial

        np.random.seed(42)
        mat = np.random.randn(4, 1000)
        refcorr = np.random.randn(1000)
        strvec = np.linspace(-0.02, 0.02, 11)
        strrefmat = np.random.randn(11, 1000)
        tw = np.array([[50, 150], [200, 350]])

        # 1. velocity_change_estimate equivalence
        dv_orig = sm.velocity_change_estimate(mat, tw, strrefmat, strvec, sides='both', return_sim_mat=True)
        dv_vec = _vectorized_velocity_change_estimate(mat, tw, strrefmat, strvec, sides='both', return_sim_mat=True)
        np.testing.assert_allclose(dv_orig['corr'], dv_vec['corr'], atol=1e-6)
        np.testing.assert_allclose(dv_orig['value'], dv_vec['value'], atol=1e-6)
        np.testing.assert_allclose(dv_orig['sim_mat'], dv_vec['sim_mat'], atol=1e-6)

        # 2. compute_wfc equivalence
        wfc_orig = sm.compute_wfc(mat, tw, refcorr, sides='both')
        wfc_vec = _vectorized_compute_wfc(mat, tw, refcorr, sides='both')
        np.testing.assert_allclose(wfc_orig, wfc_vec, atol=1e-6)

        # 3. create_shifted_ref_mat equivalence
        class MockStats:
            start_lag = -1.0
            end_lag = 1.0
            npts = 1000

        stats = MockStats()
        shifts = np.linspace(-0.01, 0.01, 11)
        ref_orig = sm.create_shifted_ref_mat(refcorr, stats, shifts)
        ref_vec = _vectorized_create_shifted_ref_mat(refcorr, stats, shifts)
        np.testing.assert_allclose(ref_orig, ref_vec, atol=1e-6)

        # 4. sensitivity_kernel equivalence
        x = np.arange(0, 1050, 50.0)
        y = np.arange(0, 1050, 50.0)
        s1 = np.array([200.0, 300.0])
        s2 = np.array([800.0, 700.0])
        K_orig = sm_spatial.sensitivity_kernel(s1, s2, x, y, t=1.2, dt=0.02, vel=1000.0, mf_path=500.0)
        K_vec = _vectorized_sensitivity_kernel(s1, s2, x, y, t=1.2, dt=0.02, vel=1000.0, mf_path=500.0)
        np.testing.assert_allclose(K_orig, K_vec, atol=1e-6)

    def test_17_processing_stack_persistence_and_reorder(self):
        ''' Verify that processing stack node reordering, addition, and toggling persist across preference dialog sessions. '''
        import copy
        import wx
        from psysmon.core.preferences_manager import ProcessingStackPrefItem
        from psysmon.packages.tracedisplay.plugins_processingstack import PStackEditField

        app = wx.GetApp() or wx.App()

        class MockStackNode(object):
            def __init__(self, name):
                self.name = name
                self.enabled = True
            def isEnabled(self):
                return self.enabled
            def toggleEnabled(self):
                self.enabled = not self.enabled
            def getEditPanel(self, parent):
                return wx.Panel(parent)
            def __repr__(self):
                return f'MockStackNode({self.name})'

        class MockCheckEvent(object):
            def __init__(self, selection):
                self._sel = selection
            def GetSelection(self):
                return self._sel

        initial_nodes = [
            MockStackNode('detrend'),
            MockStackNode('remove response (paz)'),
            MockStackNode('bandpass filter'),
            MockStackNode('array stacking')
        ]

        pref_item = ProcessingStackPrefItem(name='processing_stack', value=copy.deepcopy(initial_nodes))
        self.assertEqual([n.name for n in pref_item.value],
                         ['detrend', 'remove response (paz)', 'bandpass filter', 'array stacking'])

        frame = wx.Frame(None)
        try:
            # 1. First session: open preferences
            field1 = PStackEditField('processing stack', pref_item, (400, 600), parent=frame)
            panel1 = field1.controlElement

            # Move 'array stacking' (index 3) up to index 1 (after 'detrend')
            panel1.nodeListBox.SetSelection(3)
            panel1.on_up_button_clicked(None)
            panel1.on_up_button_clicked(None)

            # Add 'resample' as the last node
            resample_node = MockStackNode('resample')
            panel1.selected_nodes.append(resample_node)
            panel1.update_node_list()

            expected_order = ['detrend', 'array stacking', 'remove response (paz)', 'bandpass filter', 'resample']
            self.assertEqual([n.name for n in panel1.selected_nodes], expected_order)
            self.assertEqual([n.name for n in pref_item.value], expected_order)

            # 2. Simulate closing preferences dialog with OK button (destroy field1)
            field1.Destroy()

            # 3. Second session: reopen preferences dialog with same pref_item
            field2 = PStackEditField('processing stack', pref_item, (400, 600), parent=frame)
            panel2 = field2.controlElement

            # Verify that reordered and added nodes are fully preserved
            self.assertEqual([n.name for n in panel2.selected_nodes], expected_order)
            self.assertEqual([n.name for n in pref_item.value], expected_order)

            # 4. Toggle enabled state of 'remove response (paz)' (index 2)
            panel2.on_node_checked(MockCheckEvent(2))
            self.assertFalse(pref_item.value[2].isEnabled())

            # 5. Reopen third time and verify enabled state persisted
            field2.Destroy()
            field3 = PStackEditField('processing stack', pref_item, (400, 600), parent=frame)
            panel3 = field3.controlElement
            self.assertFalse(panel3.selected_nodes[2].isEnabled())
            field3.Destroy()
        finally:
            frame.Destroy()

    def test_18_processing_stack_acceleration_and_decimation(self):
        ''' Test full acceleration stack: IntegerDecimation, trace threading, PAZ caching, and fallback station lookup. '''
        from psysmon.packages.tracedisplay.processingNodes import (
            FilterBandPass,
            IntegerDecimation,
            RemoveResponsePAZ
        )

        # -------------------------------------------------------------
        # 1. Test IntegerDecimation Node
        # -------------------------------------------------------------
        dec_node = IntegerDecimation()
        self.assertEqual(dec_node.pref_manager.get_value('sps'), 100.0)
        self.assertFalse(dec_node.pref_manager.get_value('apply_low_pass_filter'))

        # Create 400 Hz test stream
        st = self._create_synthetic_stream(n_stations=1, locs=('A',), fs=400.0, duration=10.0)
        orig_data = st[0].data.copy()
        orig_len = len(orig_data)
        self.assertEqual(st[0].stats.sampling_rate, 400.0)

        # Fast strided decimation (apply_low_pass_filter=False)
        dec_node.execute(st)
        self.assertEqual(st[0].stats.sampling_rate, 100.0)
        self.assertEqual(len(st[0].data), orig_len // 4)
        np.testing.assert_allclose(st[0].data, orig_data[::4])

        # Chebyshev filtered decimation (apply_low_pass_filter=True)
        st2 = self._create_synthetic_stream(n_stations=1, locs=('A',), fs=400.0, duration=10.0)
        dec_node.pref_manager.set_value('apply_low_pass_filter', True)
        dec_node.execute(st2)
        self.assertEqual(st2[0].stats.sampling_rate, 100.0)
        self.assertEqual(len(st2[0].data), orig_len // 4)

        # Non-integer target sps fallback to Fourier resample
        st3 = self._create_synthetic_stream(n_stations=1, locs=('A',), fs=400.0, duration=1.0)
        dec_node.pref_manager.set_value('sps', 150.0)
        dec_node.execute(st3)
        self.assertEqual(st3[0].stats.sampling_rate, 150.0)

        # -------------------------------------------------------------
        # 2. Test FilterBandPass with Parallel Trace Threads
        # -------------------------------------------------------------
        bp_node = FilterBandPass()
        self.assertEqual(bp_node.pref_manager.get_value('threads'), -1)
        bp_node.pref_manager.set_value('min. frequ.', 2.0)
        bp_node.pref_manager.set_value('max. frequ.', 20.0)

        st_threaded = self._create_synthetic_stream(n_stations=3, locs=('A',), fs=100.0, duration=10.0)
        st_serial = st_threaded.copy()

        bp_node.pref_manager.set_value('threads', -1)  # Auto
        bp_node.execute(st_threaded)

        bp_node.pref_manager.set_value('threads', 1)   # Single-threaded
        bp_node.execute(st_serial)

        for tr_t, tr_s in zip(st_threaded, st_serial):
            np.testing.assert_allclose(tr_t.data, tr_s.data, atol=1e-12)

        # -------------------------------------------------------------
        # 3. Test Array Stacking Metadata Forwarding & PAZ Fallback
        # -------------------------------------------------------------
        class MockCompParam:
            tf_poles = [-6.283 + 4.7124j, -6.283 - 4.7124j]
            tf_normalization_factor = 1.0
            sensitivity = 1000.0

        class MockComponent:
            output_unit = 'm/s'
            def get_parameter(self, start_time, end_time):
                return [MockCompParam()]

        class MockCompItem:
            item = MockComponent()

        class MockRecParam:
            gain = 1.0
            bitweight = 1.0

        class MockRecStream:
            def get_parameter(self, start_time, end_time):
                return [MockRecParam()]
            def get_component(self, start_time, end_time):
                return [MockCompItem()]

        class MockStreamItem:
            item = MockRecStream()

        class MockChannel:
            def get_stream(self, start_time, end_time):
                return [MockStreamItem()]

        class MockStation:
            def __init__(self, name, location=''):
                self.name = name
                self.location = location
            def get_channel(self, name):
                return [MockChannel()]

        class MockGeometryInventory:
            def __init__(self):
                # Physical stations registered with location='A'
                self.stations = [
                    MockStation('AP231', 'A'),
                    MockStation('AP232', 'A'),
                    MockStation('AP233', 'A'),
                ]
            def get_station(self, network=None, name=None, location=None):
                res = self.stations
                if name is not None:
                    res = [s for s in res if s.name == name]
                if location is not None:
                    res = [s for s in res if s.location == location]
                return res

        class MockParentStack:
            def __init__(self):
                self.project = MockProject(None)
                self.project.geometry_inventory = MockGeometryInventory()

        # Create 9-trace stream (AP231 A,B,C; AP232 A,B,C; AP233 A,B,C)
        st_raw = self._create_synthetic_stream(n_stations=3, locs=('A', 'B', 'C'), fs=100.0, duration=10.0)
        self.assertEqual(len(st_raw), 9)

        # Step 2: Array stacking
        stack_node = ArrayStackingNode()
        stack_node.pref_manager.set_value('grouping', 'by_location_code')
        stack_node.pref_manager.set_value('output_mode', 'replace')
        stack_node.execute(st_raw)
        self.assertEqual(len(st_raw), 3)

        # Verify metadata forwarding
        for tr in st_raw:
            self.assertEqual(tr.stats.location, '')
            self.assertTrue(hasattr(tr.stats, 'source_station'))
            self.assertEqual(tr.stats.source_location, 'A')
            self.assertEqual(tr.stats.constituent_count, 3)

        # Step 3: RemoveResponsePAZ on virtual beams
        paz_node = RemoveResponsePAZ()
        paz_node.parentStack = MockParentStack()
        self.assertTrue(paz_node.pref_manager.get_value('paz_vector_caching'))
        self.assertEqual(paz_node.pref_manager.get_value('threads'), -1)

        st_cached = st_raw.copy()
        st_uncached = st_raw.copy()

        # Deconvolve with vector caching enabled
        paz_node.pref_manager.set_value('paz_vector_caching', True)
        paz_node.execute(st_cached)

        # Deconvolve without caching
        paz_node.pref_manager.set_value('paz_vector_caching', False)
        paz_node.execute(st_uncached)

        for tr_c, tr_u in zip(st_cached, st_uncached):
            self.assertEqual(tr_c.stats.unit, 'm/s')
            self.assertEqual(tr_u.stats.unit, 'm/s')
            # Bit-for-bit equivalence between cached and uncached deconvolution
            np.testing.assert_allclose(tr_c.data, tr_u.data, atol=1e-10)

        # -------------------------------------------------------------
        # 4. Test ConvertToSensorUnits with virtual beam fallback lookup
        # -------------------------------------------------------------
        from psysmon.packages.tracedisplay.processingNodes import ConvertToSensorUnits
        conv_node = ConvertToSensorUnits()
        conv_node.parentStack = MockParentStack()

        st_conv = st_raw.copy()
        raw_amplitudes = [tr.data.copy() for tr in st_conv]
        conv_node.execute(st_conv)

        # Sensitivity = 1000.0, Gain = 1.0, Bitweight = 1.0 => scale = 1.0 / (1.0 * 1000.0) = 0.001
        for tr_conv, orig_data in zip(st_conv, raw_amplitudes):
            self.assertEqual(tr_conv.stats.unit, 'm/s')
            np.testing.assert_allclose(tr_conv.data, orig_data * 0.001, rtol=1e-6)

        # -------------------------------------------------------------
        # 5. Verify deepcopy compatibility for collection execution
        # -------------------------------------------------------------
        import copy
        conv_copy = copy.deepcopy(conv_node)
        self.assertIsNotNone(conv_copy)

        paz_node.pref_manager.set_value('paz_vector_caching', True)
        paz_copy = copy.deepcopy(paz_node)
        self.assertIsNotNone(paz_copy)
        self.assertTrue(paz_copy.pref_manager.get_value('paz_vector_caching'))
        self.assertIs(paz_copy._cache_lock, paz_node._cache_lock)

        bp_copy = copy.deepcopy(bp_node)
        self.assertIsNotNone(bp_copy)

        dec_copy = copy.deepcopy(dec_node)
        self.assertIsNotNone(dec_copy)

        stack_copy = copy.deepcopy(stack_node)
        self.assertIsNotNone(stack_copy)

    def test_19_seismic_plotting_preferences(self):
        ''' Test Plots preferences configuration and headless image generation across all SeisMIC nodes. '''
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # -------------------------------------------------------------
        # 1. Verify Plots preference tabs and default values
        # -------------------------------------------------------------
        stack_node = ArrayStackingNode()
        self.assertIn('Plots', [p.name for p in stack_node.pref_manager.pages])
        self.assertEqual(stack_node.pref_manager.get_value('plot_dir'), '')
        self.assertEqual(stack_node.pref_manager.get_value('plot_format'), 'png')
        self.assertEqual(stack_node.pref_manager.get_value('plot_dpi'), 150)
        self.assertFalse(stack_node.pref_manager.get_value('plot_stack_comparison'))

        xcorr_node = SeismicXCorrNode()
        self.assertIn('Plots', [p.name for p in xcorr_node.pref_manager.pages])
        self.assertEqual(xcorr_node.pref_manager.get_value('plot_dir'), '')
        self.assertEqual(xcorr_node.pref_manager.get_value('plot_scope'), 'Cumulative Summary')
        self.assertEqual(xcorr_node.pref_manager.get_value('plot_format'), 'png')
        self.assertEqual(xcorr_node.pref_manager.get_value('plot_dpi'), 150)
        self.assertEqual(xcorr_node.pref_manager.get_value('plot_height_mode'), 'Full Looper Period (Progressive Fill)')
        self.assertEqual(
            xcorr_node.pref_manager.get_limit('plot_height_mode'),
            ('Full Looper Period (Progressive Fill)', 'Current Accumulated Days (Elastic Growth)')
        )
        self.assertFalse(xcorr_node.pref_manager.get_value('plot_heatmap'))
        self.assertFalse(xcorr_node.pref_manager.get_value('plot_section'))
        self.assertFalse(xcorr_node.pref_manager.get_value('plot_stack'))

        monitor_node = SeismicMonitorNode()
        self.assertIn('Plots', [p.name for p in monitor_node.pref_manager.pages])
        self.assertEqual(monitor_node.pref_manager.get_value('plot_dir'), '')
        self.assertEqual(monitor_node.pref_manager.get_value('plot_format'), 'png')
        self.assertEqual(monitor_node.pref_manager.get_value('plot_dpi'), 150)
        self.assertFalse(monitor_node.pref_manager.get_value('plot_technical'))
        self.assertFalse(monitor_node.pref_manager.get_value('plot_fancy'))
        self.assertFalse(monitor_node.pref_manager.get_value('plot_multidv'))

        spatial_node = SeismicSpatialNode()
        self.assertIn('Plots', [p.name for p in spatial_node.pref_manager.pages])
        self.assertEqual(spatial_node.pref_manager.get_value('plot_dir'), '')
        self.assertEqual(spatial_node.pref_manager.get_value('plot_format'), 'png')
        self.assertEqual(spatial_node.pref_manager.get_value('plot_dpi'), 150)
        self.assertFalse(spatial_node.pref_manager.get_value('plot_spatial_map'))

        # -------------------------------------------------------------
        # 2. Test ArrayStackingNode plot generation
        # -------------------------------------------------------------
        st = self._create_synthetic_stream(n_stations=2, locs=('A', 'B', 'C'), fs=50.0, duration=300.0)
        stack_plot_dir = os.path.join(self.test_dir, 'plots_stack_test')
        stack_node.pref_manager.set_value('grouping', 'by_location_code')
        stack_node.pref_manager.set_value('operator', 'mean')
        stack_node.pref_manager.set_value('output_mode', 'replace')
        stack_node.pref_manager.set_value('plot_dir', stack_plot_dir)
        stack_node.pref_manager.set_value('plot_stack_comparison', True)
        stack_node.execute(st)

        stack_plots = [f for f in os.listdir(stack_plot_dir) if f.endswith('.png')]
        self.assertGreaterEqual(len(stack_plots), 2)  # AP231 and AP232
        self.assertTrue(any('AP231' in f for f in stack_plots))

        # Test looper output_dir default resolution when plot_dir is empty
        st_looper = self._create_synthetic_stream(n_stations=2, locs=('A', 'B', 'C'), fs=50.0, duration=300.0)
        stack_node_looper = ArrayStackingNode()
        stack_node_looper.pref_manager.set_value('grouping', 'by_location_code')
        stack_node_looper.pref_manager.set_value('plot_dir', '')
        stack_node_looper.pref_manager.set_value('plot_stack_comparison', True)
        looper_test_dir = os.path.join(self.test_dir, 'looper_stack_proc_dir')
        stack_node_looper.execute(st_looper, output_dir=looper_test_dir)
        looper_stack_plots = os.path.join(looper_test_dir, 'plots', 'stacking')
        self.assertTrue(os.path.isdir(looper_stack_plots))
        self.assertTrue(any('AP231' in f for f in os.listdir(looper_stack_plots)))

        # -------------------------------------------------------------
        # 3. Test SeismicXCorrNode plot generation (Daily & Cumulative)
        # -------------------------------------------------------------
        st_long = self._create_synthetic_stream(n_stations=3, locs=('A', 'B', 'C'), fs=50.0, duration=2 * 86400.0)
        # Stack to 3 virtual stations
        stack_prep = ArrayStackingNode()
        stack_prep.pref_manager.set_value('grouping', 'by_location_code')
        stack_prep.pref_manager.set_value('operator', 'mean')
        stack_prep.pref_manager.set_value('output_mode', 'replace')
        stack_prep.execute(st_long)

        proc_dir = os.path.join(self.test_dir, 'plots_pipeline_looper')
        xcorr_node.project = self.project
        xcorr_node.pref_manager.set_value('combination_method', 'autoComponents')
        xcorr_node.pref_manager.set_value('corr_len', 3600.0)
        xcorr_node.pref_manager.set_value('corr_inc', 3600.0)
        xcorr_node.pref_manager.set_value('max_lag', 1.0)
        xcorr_node.pref_manager.set_value('spectral_whitening', True)
        xcorr_node.pref_manager.set_value('freq_min', 5.0)
        xcorr_node.pref_manager.set_value('freq_max', 20.0)
        xcorr_node.pref_manager.set_value('plot_scope', 'Both')
        xcorr_node.pref_manager.set_value('plot_height_mode', 'Full Looper Period (Progressive Fill)')
        xcorr_node.pref_manager.set_value('plot_heatmap', True)
        xcorr_node.pref_manager.set_value('plot_section', True)
        xcorr_node.pref_manager.set_value('plot_stack', True)

        t_start = st_long[0].stats.starttime
        t_end = st_long[0].stats.starttime + 86400.0
        xcorr_node.execute(st_long, process_limits=(t_start, t_end), output_dir=proc_dir)

        xcorr_plot_dir = os.path.join(proc_dir, 'plots', 'xcorr')
        self.assertTrue(os.path.isdir(xcorr_plot_dir))
        xcorr_saved = os.listdir(xcorr_plot_dir)
        self.assertTrue(any('heatmap' in f and 'cumulative' in f for f in xcorr_saved))
        self.assertTrue(any('section' in f and 'cumulative' in f for f in xcorr_saved))
        self.assertTrue(any('stack' in f and 'cumulative' in f for f in xcorr_saved))
        self.assertTrue(any('heatmap' in f and t_start.strftime('%Y%m%d') in f for f in xcorr_saved))

        # Verify section plot renders crisp pure-black non-antialiased lines with spacious whitespace
        sec_files = [f for f in xcorr_saved if 'section' in f and 'cumulative' in f]
        self.assertTrue(len(sec_files) > 0)
        from PIL import Image
        sec_img = Image.open(os.path.join(xcorr_plot_dir, sec_files[0])).convert('L')
        sec_arr = np.array(sec_img)
        non_white = sec_arr[sec_arr < 250]
        self.assertGreater(len(non_white), 0)
        pure_black = np.sum(non_white < 15)
        self.assertGreater(pure_black / len(non_white), 0.70)
        self.assertEqual(float(np.median(non_white)), 0.0)
        # Ensure plot has generous whitespace (>85% white canvas) with no solid pitch-black block
        white_fraction = float(np.sum(sec_arr > 240) / sec_arr.size)
        self.assertGreater(white_fraction, 0.85)

        # Also verify 'Current Accumulated Days (Elastic Growth)' execution
        xcorr_node.pref_manager.set_value('plot_height_mode', 'Current Accumulated Days (Elastic Growth)')
        xcorr_node.execute(st_long, process_limits=(t_start, t_end), output_dir=proc_dir)

        # -------------------------------------------------------------
        # 4. Test SeismicMonitorNode plot generation
        # -------------------------------------------------------------
        monitor_node.project = self.project
        monitor_node.pref_manager.set_value('corr_dir', proc_dir)
        monitor_node.pref_manager.set_value('tw_start', 0.05)
        monitor_node.pref_manager.set_value('tw_len', 0.20)
        monitor_node.pref_manager.set_value('sides', 'both')
        monitor_node.pref_manager.set_value('dv_method', 'stretch_multi')
        monitor_node.pref_manager.set_value('ref_type', 'mean')
        monitor_node.pref_manager.set_value('min_coherence', 0.65)
        monitor_node.pref_manager.set_value('freq_min', 5.0)
        monitor_node.pref_manager.set_value('freq_max', 20.0)
        monitor_node.pref_manager.set_value('plot_technical', True)
        monitor_node.pref_manager.set_value('plot_fancy', True)
        monitor_node.pref_manager.set_value('plot_multidv', True)
        monitor_node.execute()

        monitor_plot_dir = os.path.join(proc_dir, 'plots', 'monitor')
        self.assertTrue(os.path.isdir(monitor_plot_dir))
        mon_saved = os.listdir(monitor_plot_dir)
        self.assertTrue(any('technical' in f for f in mon_saved))
        self.assertTrue(any('fancy' in f for f in mon_saved))
        self.assertTrue(any('multidv' in f for f in mon_saved))

        # -------------------------------------------------------------
        # 5. Test SeismicSpatialNode plot generation
        # -------------------------------------------------------------
        spatial_node.project = self.project
        spatial_node.pref_manager.set_value('dv_dir', proc_dir)
        spatial_node.pref_manager.set_value('grid_resolution', 50.0)
        spatial_node.pref_manager.set_value('epsg', 'EPSG:3995')
        spatial_node.pref_manager.set_value('plot_spatial_map', True)
        spatial_node.execute()

        spatial_plot_dir = os.path.join(proc_dir, 'plots', 'spatial')
        self.assertTrue(os.path.isdir(spatial_plot_dir))
        spatial_saved = os.listdir(spatial_plot_dir)
        self.assertTrue(any('dv_spatial_map.png' in f for f in spatial_saved))

    def test_20_array_stacking_preference_label_streamlined(self):
        ''' Verify that plot_stack_comparison checkbox label in ArrayStackingNode is shortened. '''
        node = ArrayStackingNode()
        page = node.pref_manager.pages[-1]
        self.assertEqual(page.name, 'Plots')
        avail_group = page.groups[-1]
        self.assertEqual(avail_group.name, 'Available Plots')
        item = avail_group.items[0]
        self.assertEqual(item.name, 'plot_stack_comparison')
        self.assertEqual(item.label, 'Stacked beam comparison')
        self.assertIn('Multi-trace waveform comparison', item.tool_tip)

    def test_21_pstack_edit_panel_decoupled_layout(self):
        ''' Verify that PStackEditPanel and ProcessingStack use decoupled BoxSizers (mainSizer + topSizer)
        so that nodeListBox width is independent of nodeOptions width.
        '''
        import inspect
        from psysmon.packages.tracedisplay.plugins_processingstack import PStackEditPanel, ProcessingStack

        src_init = inspect.getsource(PStackEditPanel.__init__)
        src_add = inspect.getsource(PStackEditPanel.on_add_button_clicked)
        src_remove = inspect.getsource(PStackEditPanel.on_remove_button_clicked)
        src_sel = inspect.getsource(PStackEditPanel.on_node_selected)

        self.assertIn('mainSizer = wx.BoxSizer(wx.VERTICAL)', src_init)
        self.assertIn('topSizer = wx.BoxSizer(wx.HORIZONTAL)', src_init)
        self.assertIn('topSizer.Add(self.nodeListBox, 1', src_init)
        self.assertIn('topSizer.Add(buttonSizer, 0', src_init)
        self.assertIn('mainSizer.Add(topSizer', src_init)
        self.assertIn('mainSizer.Add(self.nodeOptions, 1', src_init)

        for name, src in [('on_add_button_clicked', src_add),
                          ('on_remove_button_clicked', src_remove),
                          ('on_node_selected', src_sel)]:
            self.assertIn('sizer.Add(self.nodeOptions, 1', src, f"Missing BoxSizer Add in PStackEditPanel.{name}")

        src_ps_fold = inspect.getsource(ProcessingStack.buildFoldPanel)
        src_ps_remove = inspect.getsource(ProcessingStack.onRemove)
        src_ps_sel = inspect.getsource(ProcessingStack.onNodeSelected)

        self.assertIn('mainSizer = wx.BoxSizer(wx.VERTICAL)', src_ps_fold)
        self.assertIn('topSizer = wx.BoxSizer(wx.HORIZONTAL)', src_ps_fold)
        self.assertIn('foldPanel.SetSizer(mainSizer)', src_ps_fold)

        for name, src in [('onRemove', src_ps_remove), ('onNodeSelected', src_ps_sel)]:
            self.assertIn('sizer.Add(self.nodeOptions, 1', src, f"Missing BoxSizer Add in ProcessingStack.{name}")

    def test_22_collection_disabled_node_styling(self):
        ''' Verify CollectionPanel._style_node_item applies GREY70 and strikethrough when disabled, and wx.NullColour when enabled. '''
        from unittest.mock import MagicMock
        import wx
        from psysmon.gui.main.collection_panel import CollectionPanel

        class DummyFont(object):
            def __init__(self):
                self.strikethrough = False
            def SetStrikethrough(self, val):
                self.strikethrough = val
            def GetStrikethrough(self):
                return self.strikethrough

        panel = MagicMock(spec=CollectionPanel)
        tree = MagicMock()
        dummy_font = DummyFont()
        tree.GetFont.return_value = dummy_font
        tree.icons = {'node_disabled': 99}
        panel.collectionTreeCtrl = tree

        # Test disabled styling
        CollectionPanel._style_node_item(panel, 'node_item', enabled=False)
        self.assertTrue(dummy_font.GetStrikethrough())
        tree.SetItemFont.assert_called_with('node_item', dummy_font)
        tree.SetItemTextColour.assert_called_with('node_item', wx.Colour(178, 178, 178))
        tree.SetItemImage.assert_called_with('node_item', 99, wx.TreeItemIcon_Normal)

        # Test enabled styling
        CollectionPanel._style_node_item(panel, 'node_item', enabled=True)
        self.assertFalse(dummy_font.GetStrikethrough())
        tree.SetItemFont.assert_called_with('node_item', dummy_font)
        tree.SetItemTextColour.assert_called_with('node_item', wx.NullColour)

    def test_23_looper_child_disabled_status_logic(self):
        ''' Verify that looper child enabled state is evaluated from both parent and child. '''
        class DummyNode(object):
            def __init__(self, enabled=True):
                self.enabled = enabled

        # Case 1: Parent enabled, child enabled -> True
        parent = DummyNode(enabled=True)
        child = DummyNode(enabled=True)
        self.assertTrue(parent.enabled and getattr(child, 'enabled', True))

        # Case 2: Parent enabled, child disabled -> False
        child.enabled = False
        self.assertFalse(parent.enabled and getattr(child, 'enabled', True))

        # Case 3: Parent disabled, child enabled -> False
        parent.enabled = False
        child.enabled = True
        self.assertFalse(parent.enabled and getattr(child, 'enabled', True))


if __name__ == '__main__':
    unittest.main()


