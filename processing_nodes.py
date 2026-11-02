# LICENSE
#
# This file is part of psysmon_seismic.
#
'''
Processing stack nodes for psysmon_seismic.
'''

import os
import copy
import logging
import re
import numpy as np
import psysmon
from psysmon.core.processingStack import ProcessingNode
from psysmon.core.preferences_manager import (
    SingleChoicePrefItem,
    IntegerSpinPrefItem,
    CheckBoxPrefItem,
    TextEditPrefItem,
    DirBrowsePrefItem
)

if psysmon.wx_available:
    import wx
    import wx.lib.scrolledpanel as scrolled
    import psysmon.gui.dialog.pref_listbook as psy_lb
    from psysmon.gui.bricks import PrefEditPanel, PrefPagePanel

    class ScrolledPrefEditPanel(PrefEditPanel):
        ''' Custom PrefEditPanel that wraps each notebook page in a ScrolledPanel.
        Prevents controls and dropdown windows from being clipped in constrained
        processing stack containers.
        '''
        def build_notebook(self):
            for cur_page in self.pref.pages:
                scrolled_page = scrolled.ScrolledPanel(parent=self.notebook, id=wx.ID_ANY)
                sizer = wx.BoxSizer(wx.VERTICAL)
                panel = PrefPagePanel(parent=scrolled_page, id=wx.ID_ANY, page=cur_page)
                sizer.Add(panel, 1, wx.EXPAND)
                scrolled_page.SetSizer(sizer)
                scrolled_page.SetupScrolling(scroll_x=False, scroll_y=True)
                self.notebook.AddPage(scrolled_page, cur_page.name)

logger = logging.getLogger(__name__)


class ArrayStackingNode(ProcessingNode):
    ''' Mini-array and geophone triplet stacking processing node.

    Stacks multiple geophone traces into a virtual station beam.
    Usable in both TraceDisplay and ProcessingStackLooperChild.
    '''
    nodeClass = 'common'

    def __init__(self, **kwargs):
        ProcessingNode.__init__(
            self,
            name='array stacking',
            mode='editable',
            category='array',
            tags=['array', 'stacking', 'mini-array', 'beamforming', 'seismic'],
            **kwargs
        )
        self.create_preferences()

    def create_preferences(self):
        ''' Build preferences pages and items with streamlined parameter keys. '''
        # Page 1: Grouping
        grp_page = self.pref_manager.add_page('Grouping')
        grp_group = grp_page.add_group('Grouping')
        item = SingleChoicePrefItem(
            name='grouping',
            label='grouping rule',
            value='by_location_code',
            limit=('by_location_code', 'by_station_root', 'all_to_single'),
            tool_tip=(
                'by_location_code: groups traces with same station name and differing location codes (e.g. AP231 A, B, C).\n'
                'by_station_root: groups traces whose station names share a common root (e.g. AP231A, AP231B -> AP231).\n'
                'all_to_single: stacks all traces in the stream into one single beam.'
            )
        )
        grp_group.add_item(item)

        item = TextEditPrefItem(
            name='channel',
            label='virtual channel',
            value='DPZ',
            tool_tip='Channel name for the resulting virtual stacked trace.'
        )
        grp_group.add_item(item)

        # Page 2: Stacking Operation
        op_page = self.pref_manager.add_page('Stacking')
        op_group = op_page.add_group('Stacking Operation')
        item = SingleChoicePrefItem(
            name='operator',
            label='stack operator',
            value='mean',
            limit=('mean', 'median', 'sum'),
            tool_tip='Mathematical operator to combine traces (mean = straight average).'
        )
        op_group.add_item(item)

        item = SingleChoicePrefItem(
            name='output_mode',
            label='output mode',
            value='replace',
            limit=('replace', 'replicate', 'append'),
            tool_tip=(
                'replace: replaces array members with the single stacked virtual trace (recommended for collections).\n'
                'replicate: overwrites each constituent channel with the stacked trace (recommended for TraceDisplay).\n'
                'append: keeps raw traces and appends the stacked virtual trace.'
            )
        )
        op_group.add_item(item)

        item = CheckBoxPrefItem(
            name='normalize',
            label='normalize before stack',
            value=False,
            tool_tip='Normalize each trace by its RMS amplitude before stacking.'
        )
        op_group.add_item(item)

        # Page: Plots
        plot_page = self.pref_manager.add_page('Plots')
        out_group = plot_page.add_group('Plot Output')
        item = DirBrowsePrefItem(
            name='plot_dir',
            label='plot output directory',
            value='',
            tool_tip=(
                'Custom directory to save array stacking waveform comparison plots.\n'
                'Leave empty to auto-save to <project_data_dir>/plots/stacking/.'
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
            name='plot_stack_comparison',
            label='Stacked beam comparison',
            value=False,
            tool_tip='Multi-trace waveform comparison showing constituent geophones alongside the stacked virtual beam.'
        )
        plots_group.add_item(item)

    def getEditPanel(self, parent):
        ''' Build and return the edit panel for the processing stack GUI. '''
        if psysmon.wx_available:
            return ScrolledPrefEditPanel(pref=self.pref_manager, parent=parent)
        return super().getEditPanel(parent)

    def edit(self):
        ''' Open the preferences editor dialog. '''
        if psysmon.wx_available:
            dlg = psy_lb.ListbookPrefDialog(preferences=self.pref_manager)
            dlg.ShowModal()
            dlg.Destroy()

    def execute(self, stream, process_limits=None, origin_resource=None, output_dir=None, **kwargs):
        ''' Execute the array stacking algorithm on an obspy.Stream.

        Parameters
        ----------
        stream : :class:`obspy.core.Stream`
            The stream containing constituent geophone traces.
        '''
        if not stream or len(stream) == 0:
            return

        grouping = self.pref_manager.get_value('grouping')
        stack_op = self.pref_manager.get_value('operator')
        output_mode = self.pref_manager.get_value('output_mode')
        virt_channel = self.pref_manager.get_value('channel') or 'DPZ'
        norm_rms = bool(self.pref_manager.get_value('normalize'))

        # Group traces by array key
        groups = {}
        for tr in stream:
            if grouping == 'by_location_code':
                # Group by (network, station)
                key = (tr.stats.network, tr.stats.station)
            elif grouping == 'by_station_root':
                # Match root without trailing letter (e.g. AP231A -> AP231)
                m = re.match(r'^([A-Za-z0-9]+?)[A-Za-z]$', tr.stats.station)
                root_sta = m.group(1) if m else tr.stats.station
                key = (tr.stats.network, root_sta)
            else:  # all_to_single
                key = (tr.stats.network, 'STACK')
            
            if key not in groups:
                groups[key] = []
            groups[key].append(tr)

        stacked_virtual_traces = []
        constituent_map = {}

        for (net, sta_key), traces in groups.items():
            if len(traces) == 0:
                continue

            # Check sampling rates and sample counts
            sr0 = traces[0].stats.sampling_rate
            npts_min = min(tr.stats.npts for tr in traces)
            
            # Align traces to common length using vectorized 2D matrix allocation
            data_matrix = np.array([tr.data[:npts_min] for tr in traces], dtype=np.float64)
            if norm_rms:
                rms = np.sqrt(np.mean(data_matrix**2, axis=1, keepdims=True))
                np.divide(data_matrix, rms, out=data_matrix, where=(rms > 0))

            # Apply stacking operator
            if stack_op == 'median':
                stacked_data = np.median(data_matrix, axis=0)
            elif stack_op == 'sum':
                stacked_data = np.sum(data_matrix, axis=0)
            else:  # mean
                stacked_data = np.mean(data_matrix, axis=0)

            # Create virtual stacked trace
            virt_tr = traces[0].copy()
            virt_tr.data = stacked_data
            virt_tr.stats.station = sta_key
            virt_tr.stats.location = ''
            virt_tr.stats.channel = virt_channel
            virt_tr.stats.npts = len(stacked_data)

            # Forward constituent sensor metadata for downstream PAZ deconvolution
            virt_tr.stats.source_station = traces[0].stats.station
            virt_tr.stats.source_location = traces[0].stats.location
            virt_tr.stats.source_channel = traces[0].stats.channel
            virt_tr.stats.constituent_count = len(traces)

            # Attempt to compute and update centroid coordinates
            self._update_centroid_coordinates(virt_tr, traces)

            stacked_virtual_traces.append(virt_tr)
            constituent_map[(net, sta_key)] = (traces, virt_tr)

        # Update stream according to output_mode
        if output_mode == 'replace':
            stream.traces = stacked_virtual_traces
        elif output_mode == 'replicate':
            # In replicate mode, overwrite every constituent trace with its array's stack
            for (net, sta_key), (traces, virt_tr) in constituent_map.items():
                for tr in traces:
                    tr.data = virt_tr.data.copy()
        elif output_mode == 'append':
            for v_tr in stacked_virtual_traces:
                stream.append(v_tr)

        # Generate plots according to user preferences
        self._generate_stack_plots(constituent_map, stack_op, process_limits, output_dir=output_dir)

    def _generate_stack_plots(self, constituent_map, stack_op, process_limits, output_dir=None):
        ''' Headless generation of constituent vs. stacked beam comparison plots. '''
        try:
            plot_stack_comp = bool(self.pref_manager.get_value('plot_stack_comparison'))
            if not plot_stack_comp or not constituent_map:
                return

            user_plot_dir = str(self.pref_manager.get_value('plot_dir') or '').strip()
            if user_plot_dir:
                plot_dir = user_plot_dir
            elif output_dir:
                plot_dir = os.path.join(output_dir, 'plots', 'stacking')
            elif hasattr(self, 'parentStack') and getattr(self.parentStack, 'output_dir', None):
                plot_dir = os.path.join(self.parentStack.output_dir, 'plots', 'stacking')
            else:
                parent_lc = getattr(self.parentStack, 'parent', None) if hasattr(self, 'parentStack') else None
                looper_out = getattr(parent_lc, 'output_dir', None) if parent_lc else None
                if looper_out:
                    plot_dir = os.path.join(looper_out, 'plots', 'stacking')
                else:
                    proj = None
                    if hasattr(self, 'parentStack') and self.parentStack and hasattr(self.parentStack, 'project'):
                        proj = self.parentStack.project
                    base_dir = getattr(proj, 'dataDir', None) or getattr(proj, 'data_directory', None) or './psysmon_data'
                    out_base = os.path.join(base_dir, 'output')
                    candidate = None
                    if os.path.exists(out_base):
                        looper_dirs = [os.path.join(out_base, d) for d in os.listdir(out_base) if 'time_window_looper' in d]
                        if looper_dirs:
                            candidate = max(looper_dirs, key=os.path.getmtime)
                    plot_dir = os.path.join(candidate, 'plots', 'stacking') if candidate else os.path.join(base_dir, 'plots', 'stacking')
            os.makedirs(plot_dir, exist_ok=True)

            fmt = str(self.pref_manager.get_value('plot_format') or 'png').strip().lower()
            dpi = int(self.pref_manager.get_value('plot_dpi') or 150)

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            dt_suffix = ''
            if process_limits and len(process_limits) >= 1 and process_limits[0]:
                try:
                    dt_suffix = f"_{process_limits[0].strftime('%Y%m%d')}"
                except Exception:
                    pass

            for (net, sta_key), (traces, virt_tr) in constituent_map.items():
                if not traces:
                    continue
                n_traces = len(traces) + 1
                fig, axes = plt.subplots(nrows=n_traces, ncols=1, figsize=(10, 2.2 * n_traces), sharex=True)
                if not isinstance(axes, (list, np.ndarray)):
                    axes = [axes]

                for idx, tr in enumerate(traces):
                    ax = axes[idx]
                    t = tr.times()
                    ax.plot(t, tr.data, color='dimgray', lw=0.8, label=tr.id)
                    ax.legend(loc='upper right', framealpha=0.8)
                    ax.set_ylabel("Counts")
                    ax.grid(True, linestyle=':', alpha=0.5)

                # Last subplot: virtual stacked beam
                v_ax = axes[-1]
                vt = virt_tr.times()
                v_ax.plot(vt, virt_tr.data, color='crimson', lw=1.2, label=f"{virt_tr.id} (Stacked Beam)")
                v_ax.legend(loc='upper right', framealpha=0.8)
                v_ax.set_ylabel("Counts")
                v_ax.set_xlabel("Time [s]")
                v_ax.grid(True, linestyle=':', alpha=0.5)

                axes[0].set_title(f"Array Stacking: {net}.{sta_key} ({stack_op.upper()} of {len(traces)} geophones)")
                plt.tight_layout()

                out_filename = os.path.join(plot_dir, f"stack_comparison_{sta_key}{dt_suffix}.{fmt}")
                fig.savefig(out_filename, dpi=dpi, bbox_inches='tight')
                plt.close(fig)

            logger.info("ArrayStackingNode: Stack comparison plots saved to %s", plot_dir)
        except Exception as e:
            logger.warning("ArrayStackingNode: Plot generation failed: %s", e)
        finally:
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except Exception:
                pass

    def _update_centroid_coordinates(self, virt_tr, constituent_traces):
        ''' Calculate centroid latitude, longitude, and elevation from geometry inventory.
        '''
        try:
            geom = None
            if hasattr(self, 'parentStack') and self.parentStack and hasattr(self.parentStack, 'project'):
                proj = self.parentStack.project
                if proj and hasattr(proj, 'geometry_inventory'):
                    geom = proj.geometry_inventory

            if geom is not None:
                coords = []
                for tr in constituent_traces:
                    chans = geom.get_channel(
                        station=tr.stats.station,
                        name=tr.stats.channel,
                        network=tr.stats.network,
                        location=tr.stats.location
                    )
                    if chans and len(chans) > 0:
                        stat = chans[0].parent_station
                        coords.append((getattr(stat, 'lat', 0.0), getattr(stat, 'lon', 0.0), getattr(stat, 'elev', 0.0)))
                
                if coords:
                    arr = np.array(coords, dtype=np.float64)
                    c_lat, c_lon, c_elev = np.mean(arr[:, 0]), np.mean(arr[:, 1]), np.mean(arr[:, 2])
                    if not hasattr(virt_tr.stats, 'coordinates'):
                        virt_tr.stats.coordinates = {}
                    virt_tr.stats.coordinates['latitude'] = c_lat
                    virt_tr.stats.coordinates['longitude'] = c_lon
                    virt_tr.stats.coordinates['elevation'] = c_elev
        except Exception as e:
            logger.debug("Could not resolve centroid coordinates: %s", e)


try:
    from psysmon.packages.tracedisplay.processingNodes import IntegerDecimation
except ImportError:
    pass
