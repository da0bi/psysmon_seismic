# Changelog

All notable changes to the **psysmon_seismic** (SeisMIC Interferometry Package for pSysmon) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.1] - 2026-10-30

- **Spacious Wiggle Arrangement & Daily Stacking for `xcorr_section` Plots**:
  - Eliminated trace collision and "pitch black bar" artifacts where overlapping baselines and high-amplitude oscillations formed a solid black block.
  - Reduced amplitude scaling factor to `0.40 * trace_interval` (from `1.5x`), strictly confining wiggle peaks to their vertical lane and guaranteeing at least 20% clear whitespace between adjacent traces.
  - In cumulative summary mode (`xcorr_section_*_cumulative`), automatically stacks multi-day sub-windows into **1 high-SNR daily stacked wiggle per calendar day**. Traces are spaced by 24 hours ($86,400\text{ s}$) with dynamic figure heights up to 48 inches, providing ~75–100 px of clean whitespace per trace and crystal-clear visibility across multi-month looper timelines.
- **Crisp Pure-Black Hairline Wiggles for `xcorr_section` Plots**:
  - Eliminated washed-out greyscale anti-aliasing artifacts on cross-correlation wiggle record sections (`xcorr_section_*_daily` and `xcorr_section_*_cumulative`).
  - Standardized linewidth to `0.5` pt (corresponding to 1.04 physical pixels at 150 DPI) ensuring complete contiguous raster coverage.
  - Disabled anti-aliasing (`line.set_antialiased(False)`) and explicitly enforced pure black foreground color (`line.set_color('black')`) on all wiggle line artists.
  - Generates sharp, crisp, 1-pixel pure-black hairlines (> 96% pure `#000000` pixels, 0% grey halo) with high contrast and legibility across extended multi-day looper timeframes.

## [0.3.0] - 2026-10-30

### Added
- **Dedicated 'Plots' Preference Tab Across All SeisMIC Nodes**:
  - Implemented a unified `'Plots'` preferences tab in all four SeisMIC and array nodes: `ArrayStackingNode`, `SeismicXCorrNode`, `SeismicMonitorNode`, and `SeismicSpatialNode`.
  - Configurable output directory (`plot_dir`, `DirBrowsePrefItem`), image format (`plot_format`: `png`, `pdf`, `svg`, `jpg`), and raster resolution (`plot_dpi`: `72` to `600`, default `150`).
  - Automatically resolves default directories inside the active `TimeWindowLooper` process directory (`<looper_dir>/plots/{stacking,xcorr,monitor,spatial}/`) when left blank, with seamless `output_dir` forwarding across `ProcessingStackLooperChild` and `ArrayStackingNode`.
- **Headless Non-Interactive Rendering Engine**:
  - Automatically activates Matplotlib's `Agg` non-interactive backend with guaranteed cleanup (`plt.close(fig)` / `plt.close('all')`) to eliminate GUI thread locks, memory leaks, and X11 display dependency.
  - Added robust locale fallback in `psysmon_seismic/__init__.py` protecting against missing `en_GB` locale on minimal Linux and containerized environments.
- **Node-Specific Automated Plot Generators**:
  - **`ArrayStackingNode`**: Checkbox `plot_stack_comparison` generates multi-trace waveform comparisons showing all constituent geophones alongside the stacked virtual beam.
  - **`SeismicXCorrNode`**: Checkboxes for `plot_heatmap` (2D correlation amplitude vs lag time), `plot_section` (wiggle-trace record sections), and `plot_stack` (stacked correlation waveforms).
  - **Time Looper `plot_scope` Selector**: In `SeismicXCorrNode`, user can select `'Cumulative Summary'` (multi-day master summary), `'Daily (Per Iteration)'` (timestamped 24h slices `_YYYYMMDD`), or `'Both'`.
  - **Dynamic Vertical Height Scaling (`plot_height_mode`)**:
    - Introduced `'plot_height_mode'` preference selector in `SeismicXCorrNode`:
      - `'Full Looper Period (Progressive Fill)'` (Default): Pre-allocates canvas height and locks the y-axis to the full planned looper duration ($t_{\text{start}}$ to $t_{\text{end}}$), maintaining a constant pixel-per-day resolution that fills downwards over time.
      - `'Current Accumulated Days (Elastic Growth)'`: Dynamically sizes canvas height strictly to the days processed so far, eliminating bottom whitespace during intermediate days.
    - Dynamically scales figure height: $H = \text{clamp}(2.5 + N_{\text{days}} \times 0.45, 6.0, 36.0)$ inches.
  - **Adaptive Wiggle Amplitude for `xcorr_section`**:
    - Dynamically calculates `scalingfactor = (1.5 * corr_inc) / 1e5` ($\approx 0.054$ for 1-hour sub-windows) and sets `linewidth = 0.15` pt for cumulative sections, eliminating the solid black block artifact caused by excessive 55.5-hour trace overlaps.
  - **`SeismicMonitorNode`**: Checkboxes for `plot_technical` (3-panel diagnostic plot with similarity matrix heatmap, coherence, and $\Delta v/v$), `plot_fancy` (publication-quality scatter plot colored by coherence with `inferno_r`), and `plot_multidv` (multi-station overlay with network median curve).
  - **`SeismicSpatialNode`**: Checkbox for `plot_spatial_map` (2D plan-view tomography map with inverted triangles for stations on a `seismic_r` colormap).
- **Automated Verification Suite (`tests/test_integration.py`)**:
  - Added `test_19_seismic_plotting_preferences` validating preference page schema, defaults, and headless file generation across all nodes.
  - Updated `test_08_array_stacking_gui_layout` for the 3-tab layout (`Grouping`, `Stacking`, `Plots`).
  - All 19 integration tests passing cleanly (`OK`).

## [0.2.0] - 2026-10-30

### Added
- **New Dedicated `IntegerDecimation` Node (`processingNodes.py`)**:
  - Introduced class `IntegerDecimation(ProcessingNode)` (category `'sampling'`, name `'integer decimation'`), offering ultra-fast downsampling for pre-filtered seismic streams.
  - Parameterized in samples per second (`sps`, default `100.0`, matching `Resample` user experience).
  - Added checkbox `apply_low_pass_filter` (default `False`). When `False`, executes zero-overhead contiguous strided slicing `data[::factor]`, taking only **0.054 seconds** per 24-hour trace (**160&times; faster** than Fourier sinc resampling).
  - Designed for placement after `bandpass filter`: Because a 3–25 Hz or 1–15 Hz bandpass filter already suppresses frequencies above the 50 Hz Nyquist by > 80 dB (> 10,000&times;), `apply_low_pass_filter = False` guarantees **zero aliasing** while bypassing redundant Chebyshev filtering and phase distortion.
  - Intelligent non-integer fallback: If $f_s / f_{\text{target}}$ is not an integer ratio, logs a descriptive warning and falls back to Fourier sinc resampling so processing never halts.
  - Original `Resample` node is preserved **100% untouched** for arbitrary fractional resampling rates.
- **New 'Performance' Tab in `RemoveResponsePAZ`**:
  - Added dedicated `'Performance'` page with group `'Optimization & Parallelization'`.
  - Added `'PAZ Vector Caching for identical sensors'` checkbox (`paz_vector_caching`, default `True`): Pre-evaluates and caches the high-order transfer function vector $H_{\text{inv}}(f)$ across identical sensors. Eliminates 3.84 s of polynomial evaluation per trace ($37\times$ speedup on traces 2 and 3). In multi-day looper runs, the cache persists across all days, evaluating the vector once on Day 1.
  - Added `'Parallel Trace Threads'` integer spin parameter (`threads`, default `-1` for Auto):
    - `-1` = Auto (uses `min(trace_count, cpu_cores, 4)`).
    - `1` = Single-threaded sequential loop (zero threading overhead).
    - `>1` = Explicit worker pool size.
- **Concurrent Trace Threading in `FilterBandPass`**:
  - Added `'Parallel Trace Threads'` parameter (`threads`, default `-1`) to the existing `'General'` tab.
  - Concurrently filters stream traces across CPU cores via `concurrent.futures.ThreadPoolExecutor`, taking advantage of GIL release during C-level `sosfilt`/`lfilter`.
- **Constituent Sensor Metadata Forwarding in `ArrayStackingNode` & Fallback Lookups**:
  - Forwarded `source_station`, `source_location`, `source_channel`, and `constituent_count` to virtual stacked traces (`virt_tr.stats`).
  - Implemented automatic fallback station and channel lookup in `RemoveResponsePAZ`: If the virtual beam station name (`AP231`) is not found in `geometry_inventory`, it falls back to the constituent sensor metadata (`source_station`), resolving physical calibration with zero missing-station warnings.
  - Implemented automatic fallback station and channel lookup in `ConvertToSensorUnits`: Resolves virtual beams generated by upstream `ArrayStackingNode` without unhandled `IndexError`, enabling ultra-fast scalar sensitivity conversion (~0.06 s/day) as a viable alternative for homogeneous arrays.
- **Automated Verification Suite (`tests/test_integration.py`)**:
  - Added `test_18_processing_stack_acceleration_and_decimation` verifying:
    1. Fast strided decimation vs Chebyshev filtered decimation vs Fourier fallback.
    2. Parallel trace filtering bit-for-bit equivalence with single-threaded mode (`atol=1e-12`).
    3. Early array stacking metadata forwarding and fallback inventory lookup in `RemoveResponsePAZ`.
    4. Bit-for-bit equivalence between PAZ vector cached deconvolution and standard un-cached ObsPy `simulate` (`atol=1e-10`).
    5. Virtual beam calibration and metadata fallback in `ConvertToSensorUnits`.
    6. Full `copy.deepcopy()` compatibility across all processing stack nodes (`RemoveResponsePAZ`, `FilterBandPass`, `IntegerDecimation`, `ArrayStackingNode`, `ConvertToSensorUnits`).
  - Updated `test_11_processing_stack_gui_sizing` to validate the new `'Performance'` tab in `RemoveResponsePAZ`. All 18 tests passing cleanly (`OK`).

### Fixed
- **Collection Execution `deepcopy` Error (`_thread.lock`)**:
  - Resolved `TypeError: cannot pickle '_thread.lock' object` when clicking **Execute Collection** in pSysmon (`copy.deepcopy(self.activeCollection)`).
  - Moved `_PAZ_CACHE` and `_PAZ_CACHE_LOCK` to module level in [`processingNodes.py`](file:///home/db/Software/psysmon/psysmon/lib/psysmon/packages/tracedisplay/processingNodes.py) with `@property` accessors on `RemoveResponsePAZ`, keeping node instances fully picklable and deepcopy-compatible.
  - Added unit test assertion in `test_18` verifying successful deepcopy of `RemoveResponsePAZ`, `FilterBandPass`, `IntegerDecimation`, and `ArrayStackingNode`.

### Changed
- **Appendix on Calibration Trade-Offs & Math Expression Rework (`SeisMIC_psysmon_guide.md`)**:
  - Added comprehensive Appendix covering `convert to sensor units` vs `remove response (paz)` for ambient noise interferometry, physical frequency limits for `min. frequ.` (2.0 Hz lower bound), and memory-bandwidth threading profiling.
  - Centered all primary mathematical derivations in display blocks and converted inline math expressions into clean HTML (`<i>H</i>(<i>f</i>)`, `&Delta;<i>v</i>/<i>v</i>`, `&lambda;`, etc.).

### Changed
- **Accelerated 5-Node Processing Stack Order (`SeisMIC_psysmon_guide.md`)**:
  - Updated Section 2.2.2, Tutorial 1 (Section 3), and Tutorial 2 (Section 4) to the accelerated pipeline:
    `Step 1: detrend` &rarr; `Step 2: array stacking` (early stacking 9 &rarr; 3 beams) &rarr; `Step 3: remove response (paz)` (PAZ cached + threaded) &rarr; `Step 4: bandpass filter` (threaded) &rarr; `Step 5: integer decimation` (`sps = 100.0`, `apply_low_pass_filter = False`).
  - Total daily processing stack runtime reduced from **~245 seconds down to ~6.6 seconds** (**~37&times; speedup**).
- **Version Bump**:
  - Version elevated to `0.2.0` across package metadata and documentation.

---

## [0.1.10] - 2026-10-29

### Fixed
- **Processing Stack Preferences Persistence (`plugins_processingstack.py`)**:
  - Resolved state loss bug where reordering nodes (e.g. moving `array stacking` up), adding nodes (e.g. `resample`), or toggling node checkboxes inside the preferences dialog reverted upon clicking **OK** and reopening preferences.
  - Eliminated reference decoupling in `PStackEditPanel.__init__` by writing filtered nodes back to `pref_item.value`.
  - Added `sync_pref_item(self)` to `PStackEditPanel`, ensuring that all button actions (`on_add_button_clicked`, `on_remove_button_clicked`, `on_up_button_clicked`, `on_down_button_clicked`), checklist re-renderings (`update_node_list`), and checkbox toggles (`on_node_checked`) immediately synchronize stack state with `parent.pref_item.value` and trigger registered `on_value_change` hooks.
  - Implemented the complete `Field` protocol on `PStackEditField` (`GetValue()`, `SetValue()`, `setPrefValue()`, `setControlElementValue()`, `call_hook()`) ensuring parity with standard pSysmon GUI bricks.
  - Added `GetValue()` and `SetValue()` to `PStackEditPanel` for bidirectional programmatic access.
- **Circular Import Hazard Resolution (`bricks.py`)**:
  - Replaced eager top-level module import `import psysmon.packages.tracedisplay.plugins_processingstack` and static `gui_elements['processing_stack']` dictionary assignment in `bricks.py` with dynamic/lazy registration.
  - Added self-registration in `plugins_processingstack.py` and lazy fallback in `PrefPagePanel.init_ui()`, ensuring that importing either module in any order succeeds without `AttributeError`.
- **Authoritative Processing Stack Guide Synchronization (`SeisMIC_psysmon_guide.md`)**:
  - Reordered the Master 5-Node Optimal Processing Stack in Section 2.2.2, Tutorial 1 (Section 3), and Tutorial 2 (Section 4):
    `Step 1: detrend` &rarr; `Step 2: remove response (paz)` &rarr; `Step 3: bandpass filter` &rarr; `Step 4: array stacking` &rarr; `Step 5: resample`.
  - Documented the critical inventory binding requirement: sensor calibration metadata (poles, zeros, sensitivity, bitweight) is mapped strictly to physical hardware stations in pSysmon's `geometry_inventory`. Running response deconvolution on individual physical sensors *before* stacking ensures accurate conversion from counts to physical velocity (m/s) and prevents uncalibrated virtual beam warnings (`No station found in inventory`).
- **Correlator Options Optimization (`utils.py`)**:
  - Added `'joint_norm': False` to `co` dictionary in `build_correlator_config()`, permanently eliminating SeisMIC's `UserWarning: Keywork joint_norm not found in options['co']. Assuming False.` from run logs.
- **Automated Integration Test Verification (`tests/test_integration.py`)**:
  - Added `test_17_processing_stack_persistence_and_reorder` to verify:
    1. Reordering nodes (`array stacking` moved up twice).
    2. Adding nodes (`resample` appended to stack).
    3. Closing the preferences dialog (field destroyed).
    4. Reopening the preferences dialog and verifying full preservation of new node order and contents.
    5. Toggling node enabled states and verifying persistence across reopenings.
  - All 17 integration tests passing cleanly in 15.5s with zero `joint_norm` warnings.

---

## [0.1.9] - 2026-10-29

### Added
- **Array Stacking Vectorization (`ArrayStackingNode`)**:
  - Replaced trace-by-trace iteration and float casting with 2D matrix direct allocation `np.array([tr.data[:npts_min] for tr in traces], dtype=np.float64)`.
  - Replaced scalar RMS loop with axis-1 RMS norm and in-place division: `np.divide(data_matrix, rms, out=data_matrix, where=(rms > 0))`.
  - Measured 1.32&times; speedup on 9-geophone mini-arrays with identical numerical output.
- **<i>O</i>(1) In-Memory Trace Hash Indexing (`PsysmonStoreClient`)**:
  - Implemented `self._trace_map = defaultdict(list)` in `__init__` for instantaneous constant-time lookups `{(network, station, channel): [tr, ...]}` in `_load_local()`.
  - Eliminates linear scanning through all in-memory stream traces during waveform ingestion.
- **Batched 2D Cross-Correlation Matrix Computation (`_vectorized_pxcorr_matrix`)**:
  - Precomputes trace energy norms across <i>K</i> traces once (`2.0 * np.sum(B.real**2 + B.imag**2, axis=1) - B[:, 0].real**2`), eliminating redundant <i>O</i>(<i>K</i><sup>2</sup>) energy summations.
  - Batched 2D `irfft(..., axis=1)` and vectorized lag slicing, removing per-pair interpretive Python loops.
  - Automatically injected into parallel workers (`_parallel_correlator_worker`) and scoped via `_patch_correlator_optimizations()` context manager in single-process mode.
- **Vectorized Signal Conditioning Routines**:
  - `_vectorized_clip`: In-place C clipping with broadcasting `np.clip(A, -ts, ts, out=A)`.
  - `_vectorized_detrendqr`: 2D BLAS GEMM linear detrending `coeff = data @ rq.T; data -= coeff @ X.T`.
  - `_vectorized_FDfilter`: Zero-allocation frequency domain cosine tapering via direct NumPy broadcasting `B *= tap`.
- **Ultra-Fast Sliced Stretching Inversion (`_vectorized_velocity_change_estimate`, `_vectorized_compute_wfc`)**:
  - Replaced full-matrix zero-mask tiling (`np.tile(mask, (nstr, 1))`) and 100,000-column dot products with direct active time-window slicing: `first_sub = mat[:, ctw]; second_sub = strrefmat[:, ctw]`.
  - Yields **488.8&times; local measured speedup** (402.07 ms &rarr; 0.82 ms) while maintaining bit-exact equivalence (&Delta; &lt; 10<sup>&minus;15</sup>).
  - Batched spline evaluation in `_vectorized_create_shifted_ref_mat`: `s((times[None, :] + shifts[:, None]).ravel()).reshape(len(shifts), len(times))` (1.5&times; speedup).
  - Injected via scoped `_patch_stretching_optimizations()` context manager in `SeismicMonitorNode.execute()`.
- **Batched Coordinate Reprojection & 3D Sensitivity Kernel Inversion (`SeismicSpatialNode`)**:
  - Batched station coordinate reprojection using a single C call: `transformer.transform(lons, lats)`.
  - Vectorized 2D sensitivity kernel computation (`_vectorized_sensitivity_kernel`): evaluates 3D probability density tensors over time vector `T` in `probability()` rather than looping over scalar timestamps, yielding **2.14&times; faster** spatial kernel evaluation.
  - Injected via scoped `_patch_spatial_optimizations()` in `SeismicSpatialNode.execute()`.
- **Automated Integration Test Verification (`tests/test_integration.py`)**:
  - Added `test_14_array_stacking_vectorized_equivalence` (mean, median, sum, RMS norm equivalence).
  - Added `test_15_pxcorr_matrix_and_preprocessing_vectorized_equivalence` (clip, detrendqr, FDfilter equivalence).
  - Added `test_16_stretching_and_spatial_vectorized_equivalence` (velocity_change_estimate, compute_wfc, create_shifted_ref_mat, sensitivity_kernel equivalence).
  - All 16 test suites passing cleanly in 10.2s.

---

## [0.1.8] - 2026-10-29

### Added
- **Master 5-Node Optimal Processing Stack Table (`SeisMIC_psysmon_guide.md`)**:
  - Introduced the authoritative 5-node optimal processing stack table in Section 2.2.2 featuring the renamed column **`Optimal Parameter Settings`** listing **all parameters** for each node:
    1. `detrend`: `detrend method: constant` (mean subtraction in &lt; 1 ms removing DC bias before beamforming).
    2. `array stacking`: `grouping: by_location_code`, `channel: DPZ`, `operator: mean`, `output_mode: replace`, `normalize: False` (stacks 9 geophones to 3 beams before deconvolution, slashing downstream FFT deconvolution load by 66.7%).
    3. `remove response (paz)`: `output quantity: velocity`, `water level: 60.0 dB`, `taper ends: True`, `taper fraction: 0.05`, `use pre-filter: True` (`f1=2.0`, `f2=4.0`, `f3=40.0`, `f4=48.0` Hz).
    4. `bandpass filter`: `min. frequ.: 3.0 Hz` (or 1.0 Hz for cross-correlation), `max. frequ.: 25.0 Hz` (or 15.0 Hz for cross-correlation), `corners: 4`, `zero_phase: False` (anti-aliasing prerequisite).
    5. `resample`: `sps: 100.0` (4&times; sample reduction, 4&times; faster correlation, 4&times; smaller HDF5).
  - Accompanied by complete physical, mathematical, and computational explanations in `Explanation for Optimum Settings`.
- **Pre-Correlation Resampling & Nyquist Aliasing Deep-Dive**:
  - Documented why `sps = 50.0` fails: triggers ObsPy warning `Selected high corner frequency (25.0) of bandpass is at or above Nyquist (25.0). Applying a high-pass instead`, silently dropping the lowpass filter; Butterworth -3 dB leakage causes aliasing of 25–40 Hz ambient noise; limits lag discretization to coarse 20 ms.
  - Documented why `sps = 100.0` succeeds: provides a 25 Hz guard band with zero aliasing, zero ObsPy warnings, and fine 10 ms lag resolution for body-wave reflection tracking.
- **Node Execution Order Rationale**:
  - Mathematically proved linearity of deconvolution: `Deconv((A + B + C) / 3) == (Deconv(A) + Deconv(B) + Deconv(C)) / 3`, proving stacking before deconvolution is identical while running 3&times; faster.
  - Proved bandpass filtering before Fourier resampling is the required anti-aliasing lowpass filter.
- **Hardware & Mobile CPU Thermal / Power Budget Analysis**:
  - Documented 15W–25W TDP mobile CPU thermal throttling behavior: running 8 workers triggers aggressive frequency downclocking (95&deg;C &rarr; 1.8 GHz), penalizing serial processing stack runtime by +18.3%; running 4 workers sustains boost clocks (3.2–4.0 GHz).
- **Real-World APO23 Performance Benchmarks (Section 3.5)**:
  - Added benchmark comparison table across the 189-day dataset:
    - Baseline single-core: 268.55 min (4h 28m)
    - Unoptimized 8-worker: 287.01 min (4h 47m)
    - Optimized 5-node stack (4 workers, WARNING log, 100 sps): **~45 min** (&gt; 83% overall time savings!).
- **Automated Integration Test Suite Expansion**:
  - Added `test_11_psysmon_processing_stack_gui_sizing` (dialog height &ge; 500 px, PAZ options fully visible).
  - Added `test_12_pstack_remove_and_reorder_buttons` (node removal, up/down reordering, selection tracking).
  - Added `test_13_add_node_dialog_sizing` (dialog dimensions &ge; 860x480, full column widths, zero vertical scrolling, double-click activation).

### Fixed
- **Okular Markdown Rendering Resolution (`SeisMIC_psysmon_guide.md`)**:
  - Resolved issue where Okular's Discount Markdown engine (`libmarkdown.so.2`) and Qt's `QTextDocument` collapsed Subsection 1.3 (`Storage of Results & ResultBag`) into a single giant run-on text block.
  - Enforced strict vertical blank-line spacing before lists and nested sub-lists and wrapped all path tokens (`<base_output_dir>`, `<process_dir>`) in backticks.
- **Native HTML Math Expressions**:
  - Converted all 170 inline LaTeX math expressions (`$...$`) and display blocks (`$$...$$`) to semantic HTML tags (`<i>`, `<sup>`, `<sub>`) and character entities (`&times;`, `&Delta;`, `&tau;`, `&ell;<sup>*</sup>`, `&approx;`, `&radic;`, `&plusmn;`, `&deg;`).
  - Eliminates LaTeX mangling in Discount and guarantees native rendering in Okular, Qt GUIs, web browsers, and GitHub.

---

## [0.1.7] - 2026-10-28

### Added
- **Multi-Core ProcessPoolExecutor Parallelization with Temporal Chunking (`SeismicXCorrNode`)**:
  - Added `parallel_workers` preference (`IntegerSpinPrefItem`, default `4`, limits `1`–`32`) under dedicated **Parallel Processing** preference group.
  - Divides continuous daily streams into <i>N</i> chunks along `corr_inc` boundaries, running each worker in an isolated Python subprocess to completely bypass Python's Global Interpreter Lock (GIL).
  - Isolated staging and lock-free dataset copy-merging (`h5py.File.copy()`) into primary correlation databases in &lt; 0.1 s, preventing HDF5 POSIX lock contention (`[Errno 11]`).
  - Automatic thread oversubscription protection: worker environments set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` to prevent CPU cache thrashing.
- **Zero-Disk In-Memory Waveform Injection (`PsysmonStoreClient`)**:
  - Enhanced `PsysmonStoreClient` to accept `stream` in memory, overriding `_translate_wildcards` and `_load_local` to serve trace slices directly from RAM.
  - Completely eliminates temporary miniSEED disk staging and cleanup overhead.
- **Redundant Detrending Bypass & Stack Prerequisite Logging**:
  - Intercepts and bypasses SeisMIC's `ppst.detrend_st` during correlation execution, saving &gt; 3.4 s per day of redundant least-squares computation.
  - Logs explicit guidance notice: `"SeismicXCorrNode: Redundant 24h stream detrending in SeisMIC is bypassed for performance. Ensure incoming waveforms have been detrended in the pSysmon processing stack prior to correlation."`
- **Configurable Diagnostic Logging (`log_level`)**:
  - Added `log_level` preference (`SingleChoicePrefItem`, default `'WARNING'`, options `'WARNING'`, `'INFO'`, `'DEBUG'`).
  - Suppresses routine per-trace INFO logs, reducing log bloat and CPU overhead.
- **User Guide Performance Section (`SeisMIC_psysmon_guide.md`)**:
  - Added Section 2.2.1 ("Multi-Core Parallelization & Zero-Disk Architecture").
  - Added Section 2.2.2 ("Workflow & Parameter Optimization for Maximum Speed") detailing stack detrending prerequisite, pre-correlation decimation, tailoring `max_lag`, logging level, and parallel workers.
- **Automated Test Suite Expansion**:
  - Added `test_10_seismic_xcorr_parallel_and_in_memory` in `tests/test_integration.py` verifying preferences, sequential vs. parallel dataset numerical identity, zero disk miniSEED staging, and clean worker directory cleanup.

---

## [0.1.6] - 2026-10-28

### Added
- **Detailed Physics Reference in User Guide (`SeisMIC_psysmon_guide.md`)**:
  - Added Section 2.4.1 providing an in-depth theoretical, mathematical, and practical guide for the **Scattering & Medium Physics** parameters:
    - **`wave_velocity` (<i>v</i><sub>0</sub>)**: Ballistic wave phase propagation speed, direct arrival time calculations (<i>t</i><sub>0</sub> = &Delta; / <i>v</i><sub>0</sub>), and glaciological tuning (1650.0 m/s Rayleigh wave in ice vs. 3600.0 m/s in crustal bedrock).
    - **`mean_free_path` (&ell;<sup>*</sup>)**: Transport mean free path governing the crossover from ballistic (<i>t</i> &ll; &ell;<sup>*</sup> / <i>v</i><sub>0</sub>) to diffusive multiple scattering (<i>t</i> &gg; &ell;<sup>*</sup> / <i>v</i><sub>0</sub>) with diffusion coefficient <i>D</i> = <i>v</i><sub>0</sub> &ell;<sup>*</sup> / 2, and kernel spatial broadening in heavily crevassed ice (200–500 m) vs. homogeneous ice (1000–2000 m).
    - **`corr_len` (<i>a</i>)**: Spatial correlation length for a priori model covariance regularization <i>C</i><sub><i>m</i></sub>(<i>i</i>, <i>j</i>) = &sigma;<sub><i>m</i></sub><sup>2</sup> exp(&minus;<i>d</i><sub><i>ij</i></sub> / <i>a</i>), eliminating checkerboard inversion noise across sparse station arrays.
- **Resolution Limit Correction & Auto-Sanitization in `SeismicSpatialNode`**:
  - Adjusted `grid_resolution` limits from `(0.1, 1e6)` to `(1.0, 10000.0)` with default `50.0 m`.
  - Added `_sanitize_resolution()` to automatically detect legacy values &lt; 1.0 (such as legacy `0.001` degrees or clamped `0.1` meters) in existing project collections and upgrade them to `50.0 m`.
  - Added defensive minimum bounds checking in `execute()` ensuring `res >= 1.0 m`.

---

## [0.1.5] - 2026-10-28

### Added
- **Free-Text Projected CRS Entry (`TextEditPrefItem`) in `SeismicSpatialNode`**:
  - Replaced the hardcoded 4-choice dropdown with a flexible free-text input field defaulting to `'EPSG:3995'` (WGS 84 / Arctic Polar Stereographic).
  - Users can specify any valid projected metric CRS (e.g. `EPSG:3995`, `EPSG:3413`, `EPSG:32627`).
- **Strict Error Interception & Execution Halting on Geographic CRS (Degrees)**:
  - Added `validate_metric_crs()` helper using `pyproj.CRS`.
  - In GUI preferences (`edit()`), opening and confirming a degree-based CRS (e.g. `EPSG:4326`) pops up an explicit `wx.MessageBox` error dialog alerting the user that degree-based CRS is not permitted.
  - In pipeline execution (`execute()`), if `target_crs.is_geographic` is `True`, an explicit error is logged and execution is **aborted immediately with 0 results**, preventing unphysical directional distortion in 2D sensitivity kernels.
- **Metric Coordinate Reprojection via `pyproj.Transformer`**:
  - Station coordinates in degrees (`lon, lat`) from pSysmon's `geometry_inventory` are automatically reprojected to metric Cartesian coordinates (<i>X</i>, <i>Y</i>) in meters.
  - Metric bounding box (<i>x</i><sub>min</sub>, <i>x</i><sub>max</sub>, <i>y</i><sub>min</sub>, <i>y</i><sub>max</sub>) is constructed with padding in meters.
  - Grid cell resolution is defined and calculated in meters (`resolution [m]`, default `50.0 m`).
  - Standard WKT projection strings are automatically embedded into the exported `.prj` file so GIS applications (QGIS, ArcGIS) immediately recognize the coordinate reference system.
- **Automated Integration Test `test_09`**:
  - Validates `validate_metric_crs()` against empty strings, invalid syntax, geographic CRS, and projected metric systems.
  - Validates that `SeismicSpatialNode.execute()` aborts on `EPSG:4326` and succeeds with metric coordinates on `EPSG:3995`.

---

## [0.1.4] - 2026-10-28

### Added
- **Tabbed & Scroll-Safe Preferences GUI for `ArrayStackingNode`**:
  - Reorganized preferences into two compact notebook tabs:
    - **`Grouping`**: Contains grouping rule and virtual channel settings (height: 116 px).
    - **`Stacking`**: Contains stacking operator, output mode, and normalize settings (height: 145 px).
  - Implemented `ScrolledPrefEditPanel` in `ArrayStackingNode.getEditPanel()`, wrapping each tab in a `wx.lib.scrolledpanel.ScrolledPanel` with vertical scrolling (`scroll_x=False, scroll_y=True`).
  - Resolves the GUI clipping issue where the `STACKING OPERATION` static box and its dropdown choices were cut off in the Processing Stack dialog.
- **Streamlined Parameter Keys**:
  - Replaced verbose legacy keys with clean, concise parameter keys directly:
    - `grouping_rule` &rarr; `grouping`
    - `stack_operator` &rarr; `operator`
    - `virtual_channel` &rarr; `channel`
    - `normalize_before_stack` &rarr; `normalize`
    - `output_mode` &rarr; `output_mode`
  - Dropped legacy backward compatibility shims in favor of a clean, modern design.
- **Automated Test Suite Expansion**:
  - Added `test_08_array_stacking_gui_layout` in `tests/test_integration.py` to verify tabbed pages, streamlined parameter keys, and `ScrolledPrefEditPanel` instantiation.
  - Updated existing tests to use the new streamlined parameter keys.

---

## [0.1.3] - 2026-10-28

### Added
- **Array Centroid Coordinate Calculation in In-Memory Inventory**:
  - Enhanced `psysmon_to_obspy_inventory()` to traverse pSysmon's hierarchical `p_inv.networks` &rarr; `net.stations` structure.
  - Automatically identifies mini-array geophone clusters (`AP231A`, `AP231B`, `AP231C`) and calculates virtual station array centroids (`AP231`, `AP232`, `AP233`) using arithmetic mean coordinates ((lat, lon, <i>z</i>)).
  - Enables SeisMIC's `Correlator` to automatically compute and embed geodesic distance (`dist`), forward azimuth (`az`), and back-azimuth (`baz`) into cross-correlation HDF5 headers without requiring any disk-based inventory XML files.
- **Evaluation & Documentation for Stacked and Unstacked Correlations**:
  - Added Section 2.5 ("Evaluation: Permutations of Stacked & Unstacked Correlations") to `SeisMIC_psysmon_guide.md`, detailing capabilities across all 4 modes (stacked auto, stacked cross-station, unstacked intra-array, unstacked inter-station).
  - Added two dedicated Best Practice Guides for the `apo23` alpine glacier dataset:
    - **Best Practice Guide 1**: Autocorrelation of stacked station data for local ice thickness and basal reflection echo monitoring (<i>t</i><sub>refl</sub> &approx; 0.05–0.12 s, `max_lag = 1.5 s`).
    - **Best Practice Guide 2**: Inter-station cross-correlation between stacked data across 1.875–3.703 km baselines for regional englacial velocity monitoring (<i>c</i><sub><i>R</i></sub> &approx; 1650 m/s, `max_lag = 5.0 s`), followed by 2D spatial sensitivity kernel inversion with `SeisMIC Spatial`.
- **Defensive Storage Directory Initialization**:
  - In `PsysmonStoreClient.__init__`, automatically ensures `sds_stage` directory exists before passing to `Store_Client`.
- **Automated Test Suite Expansion**:
  - Added `test_07_in_memory_inventory_centroid_and_distances` in `tests/test_integration.py` to verify in-memory inventory conversion, centroid coordinates, geodesic baseline distances, and HDF5 `dist`/`az`/`baz` attribute embedding.

---

## [0.1.2] - 2026-10-28

### Added
- **Looper Process Output Directory Redirection (`SeismicXCorrNode`)**:
  - Respects the `output_dir` passed by `TimeWindowLooper` (e.g. `<base_output_dir>/smi-up.db-psysmon-apo23-...-time_window_looper/`).
  - All correlation databases (`corr/*.h5`), runtime parameter logs, and staging SDS archives are saved into `<output_dir>/seismic/`.
  - Falls back to project data directory only for standalone runs without a looper.
- **Solely User-Specified Data Sources for `SeismicMonitorNode` & `SeismicSpatialNode`**:
  - Added dedicated **Data Source** preference tab with `DirBrowsePrefItem` (`corr_dir` and `dv_dir`) to both nodes.
  - Abolished heuristic/guessing auto-discovery; execution strictly requires the user to specify their chosen input process folder, ensuring transparency and reproducibility.
  - Flexible path resolution helpers (`resolve_correlation_dir` and `resolve_dv_dir`) seamlessly accept either the looper process folder root, its `seismic` subfolder, or directly `seismic/corr`/`seismic/dv`.
  - Output files (`dv/*.npz`, `seismic_dv_v/*.csv`, `seismic_dv_shelve/*.db`, and `seismic_dv_spatial/*.asc / .prj / .json`) are saved directly into the user-specified process folder.
- **In-Memory Inventory Translation**:
  - Translated pSysmon internal `geometry_inventory` directly into an in-memory `obspy.Inventory` via `psysmon_to_obspy_inventory()`, eliminating static XML file copying to disk.
- **Automated Test Suite Expansion**:
  - Added tests for `output_dir` redirection, user-specified data source validation, and directory resolution helpers (`test_06_directory_resolution`).

---

## [0.1.1] - 2026-10-23

### Added
- **`min_coherence` Preference for `SeismicMonitorNode`**:
  - Added editable `min_coherence` preference item (`FloatSpinPrefItem`, default `0.5`, limit `0.0`–`1.0`, step `0.05`) under the **Stretching & Inversion** tab of `SeismicMonitorNode`.
  - Added `quality_pass` boolean evaluation column to published `TableResult` (`.csv`) to distinguish valid velocity change estimates from low-coherence noise.
  - Stored `min_coherence` threshold in `ShelveResult` (`.db`) metadata dictionary.
  - Updated `build_monitor_config()` to propagate `min_coherence` into the monitoring configuration dictionary.

---

## [0.1.0] - 2026-10-21

### Added
- **Initial Release of SeisMIC External Package for pSysmon**:
  - Implemented modular external pSysmon package discovered automatically via pSysmon's `PackageManager`.
- **`ArrayStackingNode` Processing Node**:
  - Registered as `nodeClass = 'common'` for both `ProcessingStackLooperChild` and `TraceDisplay`.
  - Supports spatial mini-array geophone beamforming and trace stacking prior to cross-correlation.
  - Configurable grouping methods: `by_location_code`, `by_station_root`, `all_to_single`.
  - Configurable mathematical stack operators: `mean`, `median`, `sum`.
  - Configurable output modes: `replace` (replace members with stacked trace), `replicate` (keep members with stacked data), `append` (add stacked trace alongside members).
  - Automatic station centroid coordinate calculation from project `geometry_inventory`.
- **`SeismicXCorrNode` Collection Child Node**:
  - Looper child node running under `TimeWindowLooper`.
  - Integrates SeisMIC's `Correlator.pxcorr()` ambient noise cross-correlation engine.
  - Automatically translates pSysmon waveform streams to transient SDS miniSEED archives via `PsysmonStoreClient`.
  - Stores cross-correlation matrices into HDF5 database files.
  - Preserves user flexibility to analyze raw unstacked streams with an informative log warning when unstacked geophones are detected.
- **`SeismicMonitorNode` Collection Node**:
  - Top-level collection node executing Weaver et al. (2011) multi-reference stretching algorithm.
  - Computes time-dependent relative seismic velocity changes (&Delta;<i>v</i>/<i>v</i>) across inter-station correlation pairs.
  - Packages results into pSysmon's `ResultBag` as both tabular data (`TableResult`, CSV) and Python shelve databases (`ShelveResult`, `.db`).
- **`SeismicSpatialNode` Collection Node**:
  - Top-level collection node computing 2D regional velocity change maps using radiative transfer sensitivity kernels.
  - Exports georeferenced ESRI ASCII raster grids (`Grid2dResult`, `.asc`, `.prj` with coordinate reference system) and JSON metadata.
- **Interactive wxPython Preferences GUI (`gui.py`)**:
  - Fully customizable preference dialogs (`ListbookPrefDialog`) for all four nodes with validation for sampling rates, frequency bands, stretch ranges, and channel selection.
- **Documentation & User Guide**:
  - Comprehensive guide in `SeisMIC_psysmon_guide.md` covering architecture, parameter reference, and APO23 glacier monitoring best practice tutorial.
- **Automated Test Suite**:
  - Complete integration test suite in `tests/test_integration.py` covering stacking nodes, correlation warning logic, result publishing, and end-to-end execution.
