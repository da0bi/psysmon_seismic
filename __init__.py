# LICENSE
#
# This file is part of psysmon_seismic.
#
'''
SeisMIC passive seismic interferometry package for pSysmon.

Provides collection nodes and processing stack nodes for:
- Mini-array stacking and virtual station beamforming
- Cross-correlation and autocorrelation (Correlator)
- Velocity change monitoring via multi-reference stretching (Monitor)
- 2D sensitivity kernel spatial inversion (Spatial)
'''

name = "SeisMIC Interferometry"
version = "0.3.0"
author = "Daniel Binder"
minPsysmonVersion = "0.0.1"
description = "Passive seismic interferometry, autocorrelations, and dv/v monitoring with SeisMIC."
website = "https://github.com/petermakus/SeisMIC"
docEntryPoint = "SeisMIC_psysmon_guide.md"

collection_node_modules = [
    'seismic_xcorr',
    'seismic_monitor',
    'seismic_spatial',
]

processing_node_modules = [
    'processing_nodes',
]

# Safeguard against systems without en_GB locale installed (used by SeisMIC plot routines)
import locale
_orig_setlocale = locale.setlocale

def _safe_setlocale(category, loc=None):
    if loc is None:
        return _orig_setlocale(category)
    try:
        return _orig_setlocale(category, loc)
    except locale.Error:
        for fallback in ('C.utf8', 'en_US.utf8', '', 'C'):
            try:
                return _orig_setlocale(category, fallback)
            except locale.Error:
                continue
        return 'C'

locale.setlocale = _safe_setlocale
