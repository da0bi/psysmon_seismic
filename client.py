# LICENSE
#
# This file is part of psysmon_seismic.
#
'''
Data and geometry adapter for interfacing pSysmon projects with SeisMIC.
'''

import os
import logging
from obspy.core import Stream, Trace
from obspy.core.inventory import Inventory, Network, Station, Channel
from obspy.core.utcdatetime import UTCDateTime

from psysmon_seismic.utils import scnl_to_seed_id, seed_id_to_scnl

logger = logging.getLogger(__name__)


def stream_to_sds(stream, sds_root):
    ''' Write an obspy.Stream into an SDS directory archive structure.

    SDS Structure:
    <sds_root>/YEAR/NET/STA/CHA.D/NET.STA.LOC.CHA.D.YEAR.DOY

    Parameters
    ----------
    stream : :class:`obspy.core.Stream`
        Waveform stream to write.
    sds_root : str
        Root path of the SDS archive.
    '''
    for tr in stream:
        net = tr.stats.network or 'XX'
        sta = tr.stats.station
        loc = tr.stats.location or ''
        cha = tr.stats.channel
        
        # Split trace across day boundaries if necessary
        t_cur = tr.stats.starttime
        t_end = tr.stats.endtime
        
        while t_cur < t_end:
            year = t_cur.year
            doy = t_cur.julday
            t_next_day = UTCDateTime(year=year, julday=doy) + 86400.0
            t_slice_end = min(t_end, t_next_day)
            
            sub_tr = tr.slice(t_cur, t_slice_end)
            if sub_tr.stats.npts > 0:
                day_dir = os.path.join(sds_root, str(year), net, sta, f"{cha}.D")
                os.makedirs(day_dir, exist_ok=True)
                filename = f"{net}.{sta}.{loc}.{cha}.D.{year}.{doy:03d}"
                out_path = os.path.join(day_dir, filename)
                
                if os.path.exists(out_path):
                    # Append or merge
                    try:
                        from obspy import read
                        exist_st = read(out_path)
                        exist_st.append(sub_tr)
                        exist_st.merge(method=1, fill_value='interpolate')
                        exist_st.write(out_path, format='MSEED')
                    except Exception as e:
                        logger.warning("Error appending to existing SDS file %s: %s", out_path, e)
                        sub_tr.write(out_path, format='MSEED')
                else:
                    sub_tr.write(out_path, format='MSEED')
                    
            t_cur = t_next_day


def psysmon_to_obspy_inventory(p_inv):
    ''' Convert a pSysmon GeometryInventory object into an in-memory ObsPy Inventory.
    Supports network traversal, individual geophone stations, and automatically computes
    virtual array centroid station entries (e.g. AP231 from AP231A, AP231B, AP231C)
    so SeisMIC can compute inter-station distances and azimuths for stacked correlations.
    '''
    obs_inv = Inventory(networks=[], source='pSysmon')
    if not p_inv:
        return obs_inv

    # Collect station objects from either p_inv.networks or directly from p_inv.stations
    raw_stations = []
    if hasattr(p_inv, 'networks') and p_inv.networks:
        for net in p_inv.networks:
            net_code = getattr(net, 'name', 'AP') or 'AP'
            for sta in getattr(net, 'stations', []):
                raw_stations.append((net_code, sta))
    elif hasattr(p_inv, 'stations') and p_inv.stations:
        for sta in p_inv.stations:
            net_code = getattr(sta.parent_network, 'name', 'AP') if hasattr(sta, 'parent_network') and sta.parent_network else 'AP'
            raw_stations.append((net_code, sta))

    if not raw_stations:
        return obs_inv

    nets = {}
    array_groups = {}  # (net_code, base_station_code): list of station coords & channels

    for net_code, sta in raw_stations:
        if net_code not in nets:
            nets[net_code] = Network(code=net_code)

        try:
            lon, lat = sta.get_lon_lat()
        except Exception:
            lon, lat = getattr(sta, 'x', 0.0), getattr(sta, 'y', 0.0)
        elev = getattr(sta, 'z', 0.0)

        obs_sta = Station(
            code=sta.name,
            latitude=float(lat),
            longitude=float(lon),
            elevation=float(elev)
        )
        for cha in getattr(sta, 'channels', []):
            obs_cha = Channel(
                code=cha.name,
                location_code=getattr(cha, 'location', '') or getattr(sta, 'location', '') or '',
                latitude=float(lat),
                longitude=float(lon),
                elevation=float(elev),
                depth=0.0
            )
            obs_sta.channels.append(obs_cha)
        nets[net_code].stations.append(obs_sta)

        # Detect array sub-stations: if station ends with a letter and preceded by a digit (e.g. AP231A -> AP231)
        base_sta = sta.name
        if len(sta.name) >= 3 and sta.name[-1].isalpha() and sta.name[-2].isdigit():
            base_sta = sta.name[:-1]

        group_key = (net_code, base_sta)
        if group_key not in array_groups:
            array_groups[group_key] = []
        array_groups[group_key].append({
            'lat': float(lat),
            'lon': float(lon),
            'elev': float(elev),
            'channels': getattr(sta, 'channels', [])
        })

    # For each group with multiple sensors or where base station is not explicitly in nets[net_code].stations:
    for (net_code, base_sta), members in array_groups.items():
        existing_codes = [s.code for s in nets[net_code].stations]
        if base_sta not in existing_codes:
            avg_lat = sum(m['lat'] for m in members) / float(len(members))
            avg_lon = sum(m['lon'] for m in members) / float(len(members))
            avg_elev = sum(m['elev'] for m in members) / float(len(members))

            centroid_sta = Station(
                code=base_sta,
                latitude=avg_lat,
                longitude=avg_lon,
                elevation=avg_elev
            )
            seen_cha_codes = set()
            for m in members:
                for cha in m['channels']:
                    if cha.name not in seen_cha_codes:
                        seen_cha_codes.add(cha.name)
                        centroid_cha = Channel(
                            code=cha.name,
                            location_code='',
                            latitude=avg_lat,
                            longitude=avg_lon,
                            elevation=avg_elev,
                            depth=0.0
                        )
                        centroid_sta.channels.append(centroid_cha)
            nets[net_code].stations.append(centroid_sta)

    obs_inv.networks = list(nets.values())
    return obs_inv


import numpy as np
from seismic.trace_data.waveform import Store_Client


class PsysmonStoreClient(Store_Client):
    ''' Native SeisMIC Store_Client adapter that connects pSysmon's in-memory
    data and geometry inventory directly to SeisMIC without requiring disk I/O.
    '''

    def __init__(self, project, proj_base, sds_subdir='sds_stage', inventory=None, stream=None):
        os.makedirs(os.path.join(proj_base, sds_subdir), exist_ok=True)
        super().__init__(None, proj_base, read_only=True, sds_dir=sds_subdir)
        self.project = project
        self._cached_inventory = inventory
        self.stream = stream
        self._trace_map = {}
        if stream is not None:
            for tr in stream:
                net = tr.stats.network or 'XX'
                sta = tr.stats.station
                cha = tr.stats.channel
                key = (net, sta, cha)
                if key not in self._trace_map:
                    self._trace_map[key] = []
                self._trace_map[key].append(tr)

    def read_inventory(self):
        ''' Overrides Store_Client.read_inventory() to serve pSysmon's in-memory inventory directly. '''
        if self._cached_inventory is not None:
            return self._cached_inventory

        if self.project and hasattr(self.project, 'geometry_inventory'):
            try:
                obs_inv = psysmon_to_obspy_inventory(self.project.geometry_inventory)
                if len(obs_inv.networks) > 0:
                    self.inventory = obs_inv
                    self._cached_inventory = obs_inv
                    return obs_inv
            except Exception as e:
                logger.debug("Could not convert psysmon geometry_inventory to ObsPy: %s", e)

        # Fallback to filesystem XML if in-memory inventory is not available
        try:
            return super().read_inventory()
        except Exception:
            return Inventory()

    def _translate_wildcards(self, network: str, station: str, component: str = '?', location: str = '*'):
        ''' Return available traces matching network, station, component, and location.
        If self.stream is provided, resolves in-memory traces directly without disk globbing.
        '''
        if self.stream is not None:
            res = []
            for tr in self.stream:
                net = tr.stats.network or 'XX'
                sta = tr.stats.station
                loc = tr.stats.location or ''
                cha = tr.stats.channel
                net_match = (network in ('*', net))
                sta_match = (station in ('*', sta))
                comp_match = (component in ('?', '*') or cha.endswith(component))
                loc_match = (location in ('*', loc))
                if net_match and sta_match and comp_match and loc_match:
                    res.append([net, sta, loc, cha])
            if res:
                return np.unique(np.array(res), axis=0).tolist()
            return []
        return super()._translate_wildcards(network, station, component=component, location=location)

    def _load_local(self, network: str, station: str, location: str, channel: str,
                    starttime: UTCDateTime, endtime: UTCDateTime, attach_response: bool,
                    _check_times: bool):
        ''' Load waveform slice directly from memory using O(1) hash index if self.stream is provided. '''
        if self.stream is not None:
            sub = Stream()
            matching_traces = self._trace_map.get((network, station, channel), [])
            for tr in matching_traces:
                sl = tr.slice(starttime, endtime)
                if sl.stats.npts > 0:
                    sub.append(sl.copy())
            if len(sub) == 0:
                return None
            sub.sort(['starttime'])
            if attach_response:
                try:
                    sub.attach_response(self.read_inventory())
                except Exception:
                    pass
            return sub
        return super()._load_local(network, station, location, channel, starttime, endtime, attach_response, _check_times)
