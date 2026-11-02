# LICENSE
#
# This file is part of psysmon_seismic.
#
'''
GUI elements and dialog helpers for psysmon_seismic.
'''

import psysmon

if psysmon.wx_available:
    import wx
    import psysmon.gui.dialog.pref_listbook as psy_lb


def show_node_preferences(pref_manager, title='Preferences', size=(450, 600)):
    ''' Open the standard pSysmon tabbed Listbook preference dialog.

    Parameters
    ----------
    pref_manager : :class:`psysmon.core.preferences_manager.PreferencesManager`
    title : str
    size : tuple (width, height)
    '''
    if not psysmon.wx_available:
        return None

    dlg = psy_lb.ListbookPrefDialog(
        preferences=pref_manager,
        title=title,
        size=size
    )
    result = dlg.ShowModal()
    dlg.Destroy()
    return result
