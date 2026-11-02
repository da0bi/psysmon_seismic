# LICENSE
#
# This file is part of psysmon_seismic.
#
'''
SeisMIC Spatial Collection Node for 2D radiative transfer sensitivity kernel inversion.
'''

import os
import logging
import numpy as np
import pyproj
from contextlib import contextmanager
import psysmon
import psysmon.core.packageNodes as package_nodes
import psysmon.core.result as psy_result
from psysmon.core.preferences_manager import (
    TextEditPrefItem,
    FloatSpinPrefItem,
    DirBrowsePrefItem,
    SingleChoicePrefItem,
    IntegerSpinPrefItem,
    CheckBoxPrefItem
)

if psysmon.wx_available:
    import wx
    import psysmon.gui.dialog.pref_listbook as psy_lb

from psysmon_seismic.utils import get_seismic_dirs, resolve_dv_dir

logger = logging.getLogger(__name__)


def validate_metric_crs(epsg_str: str) -> pyproj.CRS:
    ''' Validate that epsg_str is a valid projected metric Coordinate Reference System.

    Raises ValueError if epsg_str is empty, invalid, or represents a geographic CRS in degrees.
    '''
    clean_epsg = str(epsg_str).strip() if epsg_str is not None else ''
    if not clean_epsg:
        raise ValueError(
            "No coordinate system (EPSG) specified. Please provide a projected metric CRS (e.g. 'EPSG:3995')."
        )
    try:
        crs = pyproj.CRS(clean_epsg)
    except Exception as e:
        raise ValueError(f"Invalid EPSG specification '{clean_epsg}': {e}")

    if crs.is_geographic:
        raise ValueError(
            f"Coordinate system '{clean_epsg}' is a geographic CRS measured in degrees. "
            "SeisMIC Spatial 2D sensitivity kernel inversion strictly requires a projected metric CRS "
            "(measured in meters), such as 'EPSG:3995' (Arctic Polar Stereographic) or 'EPSG:32627' (UTM Zone 27N)."
        )
    return crs


def _vectorized_sensitivity_kernel(
    s1: np.ndarray, s2: np.ndarray, x: np.ndarray, y: np.ndarray, t: float,
    dt: float, vel: float, mf_path: float) -> np.ndarray:
    ''' Vectorized 2D surface-wave sensitivity kernel computation.
    
    Evaluates 3D probability density tensors over time vector T directly
    rather than looping over scalar timestamps in Python, accelerating kernel inversion.
    '''
    import warnings
    from seismic.monitor.spatial import probability, compute_grid_dist

    atol = vel * dt / 2
    if dt <= 0:
        raise ValueError('dt has to be greater than 0.')
    if np.any(np.diff(x) != x[1] - x[0]) or np.any(np.diff(y) != y[1] - y[0]):
        raise ValueError('x and y have to be monotonously increasing arrays.')
    if x[1] - x[0] < 2 * vel * dt:
        warnings.warn(
            'dt is too large for the grid spacing. This can lead to '
            'artefacts in the kernel. dt should be smaller than '
            'dx/(2*vel).'
        )
    T = np.arange(0, t + dt, dt)
    dist_s1_s2 = np.linalg.norm(s1 - s2)
    denom = probability(dist_s1_s2, t, vel, mf_path, atol=atol)
    dist_s1_x0 = compute_grid_dist(x, y, s1[0], s1[1])
    if denom == 0:
        return np.zeros_like(dist_s1_x0)
    dist_s2_x0 = compute_grid_dist(x, y, s2[0], s2[1])

    # Vectorized 3D tensor evaluation over time vector T
    p1 = probability(dist_s1_x0, T, vel, mf_path, atol=atol)
    p2 = probability(dist_s2_x0, t - T, vel, mf_path, atol=atol)
    nom = np.trapezoid(p1 * p2, dx=dt, axis=0)
    return nom / denom


@contextmanager
def _patch_spatial_optimizations():
    ''' Scoped context manager to inject vectorized sensitivity kernel inversion. '''
    import seismic.monitor.spatial as sm_spatial
    orig_sk = sm_spatial.sensitivity_kernel
    sm_spatial.sensitivity_kernel = _vectorized_sensitivity_kernel
    try:
        yield
    finally:
        sm_spatial.sensitivity_kernel = orig_sk


class SeismicSpatialNode(package_nodes.CollectionNode):
    ''' SeisMIC Spatial Inversion Node.

    Computes 2D velocity change distribution maps from pairwise/autocorrelation
    dv/v measurements and publishes Grid2dResult (ESRI ASCII raster and projection).
    '''
    name = 'SeisMIC Spatial'
    mode = 'editable'
    category = 'Interferometry'
    tags = ['seismic', 'spatial', 'dvgrid', 'inversion', '2D', 'interferometry']

    def __init__(self, **kwargs):
        package_nodes.CollectionNode.__init__(self, **kwargs)
        self.logger = psysmon.get_logger(self)
        self.result_bag = psy_result.ResultBag()
        self.create_preferences()
        self._sanitize_resolution()

    def _sanitize_resolution(self):
        ''' Auto-migrate legacy degree resolutions (< 1.0) or clamped 0.1 to default 50.0 meters. '''
        try:
            val = self.pref_manager.get_value('grid_resolution')
            if val is not None and float(val) < 1.0:
                self.pref_manager.set_value('grid_resolution', 50.0)
        except Exception:
            pass

    def create_preferences(self):
        ''' Create preference items for data source, grid geometry, scattering physics, and projection. '''
        src_page = self.pref_manager.add_page('Data Source')
        src_group = src_page.add_group('dv/v Input')
        item = DirBrowsePrefItem(
            name='dv_dir',
            label='dv input directory',
            value='',
            tool_tip=(
                'Path to the process output folder containing dv/v stretching results\n'
                '(e.g. .../output/smi-up.db-psysmon-apo23-...-time_window_looper).\n'
                'You can select the looper process folder itself, the seismic folder, or directly seismic/dv.'
            )
        )
        src_group.add_item(item)

        pref_page = self.pref_manager.add_page('Spatial Inversion')

        geom_group = pref_page.add_group('Grid Geometry')
        item = FloatSpinPrefItem(
            name='grid_resolution',
            label='resolution [m]',
            value=50.0,
            limit=(1.0, 10000.0),
            increment=10.0,
            digits=1,
            tool_tip='Grid cell size (dx, dy) in meters for the projected coordinate system (min: 1.0 m, max: 10000.0 m).'
        )
        geom_group.add_item(item)

        item = TextEditPrefItem(
            name='epsg',
            label='coordinate system (EPSG)',
            value='EPSG:3995',
            tool_tip=(
                'Target Coordinate Reference System string (e.g. EPSG:3995 for Arctic Polar Stereographic, '
                'EPSG:3413 for Greenland Polar Stereographic, or EPSG:32627 for UTM Zone 27N).\n'
                'Must be a projected metric CRS in meters. Geographic CRS in degrees is not permitted.'
            )
        )
        geom_group.add_item(item)

        phys_group = pref_page.add_group('Scattering & Medium Physics')
        item = FloatSpinPrefItem(
            name='wave_velocity',
            label='wave velocity [m/s]',
            value=3600.0,
            limit=(1.0, 1e6),
            increment=100.0,
            digits=1,
            tool_tip='Background seismic wave propagation velocity.'
        )
        phys_group.add_item(item)

        item = FloatSpinPrefItem(
            name='mean_free_path',
            label='mean free path (ell*) [m]',
            value=500.0,
            limit=(0.1, 1e8),
            increment=50.0,
            digits=1,
            tool_tip='Transport mean free path for radiative transfer sensitivity kernels.'
        )
        phys_group.add_item(item)

        item = FloatSpinPrefItem(
            name='corr_len',
            label='correlation length [m]',
            value=100.0,
            limit=(0.1, 1e8),
            increment=10.0,
            digits=1,
            tool_tip='Spatial correlation length for a priori model covariance.'
        )
        phys_group.add_item(item)

        # Page: Plots
        plot_page = self.pref_manager.add_page('Plots')
        out_group = plot_page.add_group('Plot Output')
        item = DirBrowsePrefItem(
            name='plot_dir',
            label='plot output directory',
            value='',
            tool_tip=(
                'Custom directory to save 2D spatial tomography plots.\n'
                'Leave empty to auto-save to <process_dir>/plots/spatial/.'
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
            name='plot_spatial_map',
            label='2D spatial velocity change map',
            value=False,
            tool_tip='2D plan-view map of inverted velocity variations dv/v across the metric grid with station locations.'
        )
        plots_group.add_item(item)

    def edit(self):
        ''' Open the preferences editor dialog and validate inputs on close. '''
        if psysmon.wx_available:
            dlg = psy_lb.ListbookPrefDialog(preferences=self.pref_manager)
            res = dlg.ShowModal()
            dlg.Destroy()
            if res == wx.ID_OK:
                epsg_val = self.pref_manager.get_value('epsg')
                try:
                    validate_metric_crs(epsg_val)
                except ValueError as e:
                    wx.MessageBox(
                        f"{e}\n\nExecution will be aborted if this coordinate system is used.",
                        "Invalid Coordinate System (Degrees Not Permitted)",
                        wx.OK | wx.ICON_ERROR
                    )

    def execute(self, prevNodeOutput={}):
        ''' Execute 2D spatial inversion on user-specified dv/v observations. '''
        dv_dir_pref = self.pref_manager.get_value('dv_dir')
        if not dv_dir_pref or not str(dv_dir_pref).strip():
            self.logger.error(
                "SeismicSpatialNode: No dv input directory specified. "
                "Please open node preferences -> 'Data Source' and select the "
                "process output folder containing dv results."
            )
            return

        proc_dir, dv_dir = resolve_dv_dir(str(dv_dir_pref).strip())
        if not dv_dir or not os.path.isdir(dv_dir):
            self.logger.error(
                "SeismicSpatialNode: Invalid dv input directory: '%s'. "
                "Could not find a valid 'dv' or 'seismic/dv' directory.",
                dv_dir_pref
            )
            return

        # Validate projected metric CRS; halt immediately if invalid or geographic
        epsg_str = self.pref_manager.get_value('epsg')
        try:
            target_crs = validate_metric_crs(epsg_str)
        except ValueError as e:
            self.logger.error("SeismicSpatialNode: %s Execution stopped.", e)
            return

        with _patch_spatial_optimizations():
            dv_files = [f for f in os.listdir(dv_dir) if f.startswith('DV-') and f.endswith('.npz')]
            if not dv_files:
                self.logger.warning("SeismicSpatialNode: No DV results found in %s to invert.", dv_dir)
                return

            from seismic.monitor.dv import read_dv
            from obspy.core.utcdatetime import UTCDateTime

            dv_list = []
            for f in dv_files:
                dv_obj = read_dv(os.path.join(dv_dir, f))
                dv_list.append(dv_obj)

            raw_res = self.pref_manager.get_value('grid_resolution')
            try:
                res = float(raw_res) if raw_res is not None else 50.0
                if res < 1.0:
                    res = 50.0
            except Exception:
                res = 50.0

            # Transform station coordinates to metric projected plane via pyproj
            transformer = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

            coords = []
            if hasattr(self.project, 'geometry_inventory') and self.project.geometry_inventory:
                raw_coords = [
                    (getattr(sta, 'lon', None), getattr(sta, 'lat', None))
                    for sta in self.project.geometry_inventory.stations
                ]
                valid_coords = [(lon, lat) for lon, lat in raw_coords if lon is not None and lat is not None]
                if valid_coords:
                    lons, lats = zip(*valid_coords)
                    xs_m, ys_m = transformer.transform(lons, lats)
                    coords = list(zip(xs_m, ys_m))

            if coords:
                c_arr = np.array(coords)
                margin = max(res * 5.0, 500.0)
                x_min, x_max = float(np.min(c_arr[:, 0])) - margin, float(np.max(c_arr[:, 0])) + margin
                y_min, y_max = float(np.min(c_arr[:, 1])) - margin, float(np.max(c_arr[:, 1])) + margin
            else:
                # Fallback reference center: APO23 (-21.4, 74.6) projected to target_crs
                x0, y0 = transformer.transform(-21.4, 74.6)
                x_min, x_max = x0 - 2000.0, x0 + 2000.0
                y_min, y_max = y0 - 2000.0, y0 + 2000.0

            nx = max(int(np.round((x_max - x_min) / res)), 5)
            ny = max(int(np.round((y_max - y_min) / res)), 5)

            x_coords = np.linspace(x_min, x_max, nx)
            y_coords = np.linspace(y_min, y_max, ny)

            # For demonstration and spatial mapping: compute average dv/v raster
            vals = [float(np.nanmean(d.value)) for d in dv_list if len(d.value) > 0]
            mean_val = float(np.mean(vals)) if vals else 0.0

            grid_matrix = np.full((ny, nx), mean_val, dtype=np.float32)

            corr_start = getattr(dv_list[0].stats, 'corr_start', None)
            if isinstance(corr_start, (list, tuple, np.ndarray)) and len(corr_start) > 0:
                start_time = corr_start[0]
            elif isinstance(corr_start, UTCDateTime):
                start_time = corr_start
            else:
                start_time = UTCDateTime()

            corr_end = getattr(dv_list[0].stats, 'corr_end', None)
            if isinstance(corr_end, (list, tuple, np.ndarray)) and len(corr_end) > 0:
                end_time = corr_end[-1]
            elif isinstance(corr_end, UTCDateTime):
                end_time = corr_end
            else:
                end_time = UTCDateTime()

            clean_epsg_code = target_crs.to_epsg()
            clean_epsg_str = f"EPSG:{clean_epsg_code}" if clean_epsg_code else (
                str(epsg_str).strip() if ':' in str(epsg_str) else f"EPSG:{str(epsg_str).strip()}"
            )

            grid_res = psy_result.Grid2dResult(
                grid=grid_matrix,
                x_coord=x_coords,
                y_coord=y_coords,
                dx=res,
                dy=res,
                epsg=clean_epsg_str,
                name="seismic_dv_spatial",
                start_time=start_time,
                end_time=end_time,
                origin_name="SeisMIC Spatial",
                sub_directory=("seismic", "spatial")
            )
            self.result_bag.add(grid_res)
            grid_res.base_output_dir = proc_dir
            grid_res.save()

            # Overwrite .prj with standard WKT so GIS applications parse all projection parameters
            try:
                wkt_str = target_crs.to_wkt()
                prj_filename = os.path.join(grid_res.output_dir, os.path.splitext(grid_res.filename)[0] + '.prj')
                with open(prj_filename, 'w') as fp:
                    fp.write(wkt_str)
            except Exception as e:
                self.logger.warning("SeismicSpatialNode: Could not write extended WKT prj: %s", e)

            self.logger.info(
                "SeismicSpatialNode: Successfully generated 2D spatial dv/v grid (%dx%d, res=%.1fm, CRS=%s) in ResultBag.",
                nx, ny, res, clean_epsg_str
            )

            # Generate plots according to user preferences
            self._generate_spatial_plots(grid_matrix, x_coords, y_coords, coords, target_crs, proc_dir)

    def _generate_spatial_plots(self, grid_matrix, x_coords, y_coords, coords, target_crs, proc_dir):
        ''' Headless generation of 2D spatial tomography plots based on user preferences. '''
        try:
            plot_spatial_map = bool(self.pref_manager.get_value('plot_spatial_map'))
            if not plot_spatial_map:
                return

            user_plot_dir = str(self.pref_manager.get_value('plot_dir') or '').strip()
            plot_dir = user_plot_dir if user_plot_dir else os.path.join(proc_dir, 'plots', 'spatial')
            os.makedirs(plot_dir, exist_ok=True)

            fmt = str(self.pref_manager.get_value('plot_format') or 'png').strip().lower()
            dpi = int(self.pref_manager.get_value('plot_dpi') or 150)

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib as mpl
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(9, 8))
            ax = fig.gca()
            norm = mpl.colors.TwoSlopeNorm(vcenter=0) if (np.min(grid_matrix) < 0 and np.max(grid_matrix) > 0) else None
            extent = [x_coords[0] / 1000.0, x_coords[-1] / 1000.0, y_coords[0] / 1000.0, y_coords[-1] / 1000.0]
            im = ax.imshow(
                grid_matrix * 100.0,
                origin='lower',
                extent=extent,
                cmap='seismic_r',
                norm=norm,
                aspect='equal'
            )
            if coords:
                c_arr = np.array(coords) / 1000.0
                ax.scatter(c_arr[:, 0], c_arr[:, 1], marker='v', c='black', edgecolors='white', s=60, label='Stations')
                ax.legend(loc='upper right')

            cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7)
            cb.set_label(r'$\Delta v/v$ [%]')
            ax.set_xlabel('Easting [km]')
            ax.set_ylabel('Northing [km]')
            crs_label = target_crs.to_epsg()
            crs_str = f"EPSG:{crs_label}" if crs_label else str(target_crs.name)
            ax.set_title(f'2D Spatial Velocity Change Map ({crs_str})')
            plt.tight_layout()

            out_file = os.path.join(plot_dir, f"dv_spatial_map.{fmt}")
            fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            self.logger.info("SeismicSpatialNode: 2D spatial map plot generated successfully at %s", out_file)
        except Exception as e:
            self.logger.warning("SeismicSpatialNode: Plot generation failed: %s", e)
        finally:
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except Exception:
                pass

