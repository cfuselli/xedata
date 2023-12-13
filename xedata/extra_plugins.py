import strax
import straxen

# Define here the new plugins that you want to be registered

import strax
import numpy as np
import strax
import numpy as np
import numba
import straxen
import itertools


export, __all__ = strax.exporter()

@export
class EventBasicsMulti(strax.Plugin):
    """
    Compute:
    - peak properties
    - peak positions
    of the first three main (in area) S1 and ten S2.
    
    The standard PosRec algorithm and the three different PosRec algorithms (mlp, gcn, cnn)
    are given for the S2s.
    """
        
    __version__ = '5.0.0'
    
    depends_on = ('events',
                  'peak_basics',
                  'peak_positions',
                  'peak_proximity')
    
    # TODO change name
    provides = 'event_basics_multi'
    data_kind = 'events'
    loop_over = 'events'
    
    max_n_s1 = straxen.URLConfig(default=3, infer_type=False,
                                    help='Number of S1s to consider')

    max_n_s2 = straxen.URLConfig(default=10, infer_type=False,
                                    help='Number of S2s to consider')


    peak_properties = (
        # name                dtype       comment
        ('time',              np.int64,   'start time since unix epoch [ns]'),
        ('center_time',       np.int64,   'weighted center time since unix epoch [ns]'),
        ('endtime',           np.int64,   'end time since unix epoch [ns]'),
        ('area',              np.float32, 'area, uncorrected [PE]'),
        ('n_channels',        np.int32,   'count of contributing PMTs'),
        ('n_competing',       np.float32, 'number of competing PMTs'),
        ('max_pmt',           np.int16,   'PMT number which contributes the most PE'),
        ('max_pmt_area',      np.float32, 'area in the largest-contributing PMT (PE)'),
        ('range_50p_area',    np.float32, 'width, 50% area [ns]'),
        ('range_90p_area',    np.float32, 'width, 90% area [ns]'),
        ('rise_time',         np.float32, 'time between 10% and 50% area quantiles [ns]'),
        ('area_fraction_top', np.float32, 'fraction of area seen by the top PMT array')
        )

    pos_rec_labels = ['cnn', 'gcn', 'mlp']

    def setup(self):

        self.posrec_save = [(xy + algo, xy + algo) for xy in ['x_', 'y_'] for algo in self.pos_rec_labels] # ???? 
        self.to_store = [name for name, _, _ in self.peak_properties]

    def infer_dtype(self):
                
        # Basic event properties  
        basics_dtype = []
        basics_dtype += strax.time_fields
        basics_dtype += [('n_peaks', np.int32, 'Number of peaks in the event'),]

        # For S1s and S2s
        for p_type in [1, 2]:
            if p_type == 1:
                max_n = self.max_n_s1
            if p_type == 2:
                max_n = self.max_n_s2
            for n in range(max_n):
                # Peak properties
                for name, dt, comment in self.peak_properties:
                    basics_dtype += [(f's{p_type}_{name}_{n}', dt, f'S{p_type}_{n} {comment}'), ]                

                if p_type == 2:
                    # S2 Peak positions
                    for algo in self.pos_rec_labels:
                        basics_dtype += [(f's2_x_{algo}_{n}', 
                                          np.float32, f'S2_{n} {algo}-reconstructed X position, uncorrected [cm]'),
                                         (f's2_y_{algo}_{n}',
                                          np.float32, f'S2_{n} {algo}-reconstructed Y position, uncorrected [cm]')]

        return basics_dtype

    @staticmethod
    def set_nan_defaults(buffer):
        """
        When constructing the dtype, take extra care to set values to
        np.Nan / -1 (for ints) as 0 might have a meaning
        """
        for field in buffer.dtype.names:
            if np.issubdtype(buffer.dtype[field], np.integer):
                buffer[field][:] = -1
            else:
                buffer[field][:] = np.nan

    @staticmethod
    def get_largest_sx_peaks(peaks,
                             s_i,
                             number_of_peaks=2):
        """Get the largest S1/S2. For S1s allow a min coincidence and max time"""
        # Find all peaks of this type (S1 or S2)

        s_mask = peaks['type'] == s_i
        selected_peaks = peaks[s_mask]
        s_index = np.arange(len(peaks))[s_mask]
        largest_peaks = np.argsort(selected_peaks['area'])[-number_of_peaks:][::-1]
        return selected_peaks[largest_peaks], s_index[largest_peaks]
            

    def compute(self, events, peaks):

        result = np.zeros(len(events), dtype=self.dtype)
        self.set_nan_defaults(result)

        split_peaks = strax.split_by_containment(peaks, events)

        result['time'] = events['time']
        result['endtime'] = events['endtime']

        for event_i, _ in enumerate(events):

            peaks_in_event_i = split_peaks[event_i]

            largest_s1s, s1_idx = self.get_largest_sx_peaks(peaks_in_event_i, s_i=1, number_of_peaks=self.max_n_s1)
            largest_s2s, s2_idx = self.get_largest_sx_peaks(peaks_in_event_i, s_i=2, number_of_peaks=self.max_n_s2)

            for i, p in enumerate(largest_s1s): 
                for prop in self.to_store:
                    result[event_i][f's1_{prop}_{i}'] = p[prop]

            for i, p in enumerate(largest_s2s):
                for prop in self.to_store:
                    result[event_i][f's2_{prop}_{i}'] = p[prop]  
                for name_alg in self.posrec_save:
                    result[event_i][f's2_{name_alg[0]}_{i}'] = p[name_alg[1]]

        return result




@export
class BiPo214Matching(strax.Plugin):
    """Plugin for matching S2 signals reconstructed as bismuth or polonium peaks in a dataset
    containing BiPo214 events.

    Provides:
    --------
    bi_po_214_matching : numpy.array
        Array containing the indices of the S2 signals reconstructed as bismuth or polonium peaks.
        The index is -9 if the S2 signal is not found, -2 if multiple S2 signals are found,
        or the index of the S2 signal if only one is found.

    Configuration:
    -----------
    tol : int
        Time tolerance window in ns to match S2 signals to BiPo214 events. Default is 3000 ns.
    """
    
    depends_on=('event_basics_multi', )   
    provides = 'bi_po_214_matching'

    __version__ = "2.1.7"
        
    def infer_dtype(self): 
        
        dtype = strax.time_fields + [
                (f's2_bi_match', np.int,
                 f'Index of the S2 reconstructed as a Bismuth peak'),
                (f's2_po_match', np.int,
                 f'Index of the S2 reconstructed as a Polonium peak'),
                (f'n_incl_peaks_s2', np.int,
                 f'S2s considered in the combinations, that passed the mask requirements')]
        
        return dtype

    def setup(self):
        
        self.tol = 3000 # 2 mus tolerance window
        
    def compute(self, events):
        result = np.zeros(len(events), dtype=self.dtype)
        result['time'] = events['time']
        result['endtime'] = events['endtime']
        result['s2_bi_match'] = -9
        result['s2_po_match'] = -9
        
        
        mask = self.box_mask(events)
        
        s2_bi_match, s2_po_match, n_s2s = self.find_match(events[mask], self.tol)
        
        result['s2_bi_match'][mask] = s2_bi_match
        result['s2_po_match'][mask] = s2_po_match
        result['n_incl_peaks_s2'][mask] = n_s2s
        
        return result
        
        
    def find_match(self, events, tol):
        

        # Compute the time difference between the two S1s
        dt_s1 = events['s1_center_time_0'] - events['s1_center_time_1']  
        
        dt_s1_lower = dt_s1 - tol
        dt_s1_upper = dt_s1 + tol
        dt_s1_lower[dt_s1_lower < 0] = 0
        
        # Prepare arrays to store matched S2s
        s2_bi_match = np.full(len(events), -1)
        s2_po_match = np.full(len(events), -1)
        n_s2s       = np.full(len(events), 0)
        
        # Find matching S2s
        for i, event in enumerate(events):
            # Create a list of possible S2 pairs to match
            s2s_idx = self.consider_s2s(event)
            n_s2s[i] = len(s2s_idx)
            possible_pairs = list(itertools.combinations(s2s_idx,2))

            for pair in possible_pairs:
                
                p0, p1 = str(pair[0]), str(pair[1])
                t0, t1 = event['s2_center_time_' + p0], event['s2_center_time_' + p1]
                
                dt_s2 = abs(t0 - t1)
                if dt_s1_lower[i] <= dt_s2 <= dt_s1_upper[i]:
                    if s2_bi_match[i] == -1:
                        s2_bi_match[i] = pair[np.argmin([t0, t1])]
                        s2_po_match[i] = pair[np.argmax([t0, t1])]
                    else:
                        s2_bi_match[i] = -2
                        s2_po_match[i] = -2
                        break
        
        return s2_bi_match, s2_po_match, n_s2s

    @staticmethod
    def consider_s2s(event):
        res = []
        for ip in range(10):
            p = '_'+str(ip)
            consider = True
            consider &= event['s2_area'+p] > 1500                                # low area limit
            consider &= event['s2_area_fraction_top'+p] > 0.5                    # to remove S1 afterpulses
            consider &= np.abs(event['s2_time'+p] - event['s1_time_0']) > 1000  # again to remove afterpulses
            consider &= np.abs(event['s2_time'+p] - event['s1_time_1']) > 1000  # and again to remove afterpulses
            consider &= event['s2_time'+p]-event['s1_time_0'] < 5000000         # 5000mus, S2 is too far in time, not related to Po
            if consider:
                res.append(ip)
        return res

    @staticmethod
    def box_mask(events):
        
        # s1_0 == alpha (Po)
        mask =  (events['s1_area_0'] >  40000) # this is implicit from band cut
        mask &= (events['s1_area_0'] < 120000)    
        mask &= (events['s1_area_fraction_top_0'] < 0.6)

        return mask

@export
class EventBiPoBasics(straxen.EventBasics):
    """
    Carlo explain please
    """
        
    __version__ = '1.0.0'
    
    depends_on = ('events',
                  'event_basics_multi',
                  'bi_po_214_matching',
                  'peak_basics',
                  'peak_positions',
                  'peak_proximity')
    
    # TODO change name
    provides = 'event_basics'
    data_kind = 'events'
    loop_over = 'events'
    

    def fill_events(self, result_buffer, events, split_peaks):
        """Loop over the events and peaks within that event"""
        for event_i, _ in enumerate(events):
            peaks_in_event_i = split_peaks[event_i]
            n_peaks = len(peaks_in_event_i)
            result_buffer[event_i]['n_peaks'] = n_peaks

            if not n_peaks:
                raise ValueError(f'No peaks within event?\n{events[event_i]}')

            self.fill_result_i(result_buffer[event_i], peaks_in_event_i, _['s2_bi_match'], _['s2_po_match'])


    def fill_result_i(self, event, peaks, bi_i, po_i):
            """For a single event with the result_buffer"""

            if (bi_i>=0) and (po_i>=0):

                largest_s2s, s2_idx = self.get_largest_sx_peaks(peaks, s_i=2, number_of_peaks=0)

                bipo_mask = [bi_i, po_i]
                s2_idx = s2_idx[bipo_mask]
                largest_s2s = largest_s2s[bipo_mask]

                largest_s1s, s1_idx = self.get_largest_sx_peaks(
                    peaks,
                    s_i=1,
                    number_of_peaks=2)

                largest_s1s = largest_s1s[::-1]
                s1_idx = s1_idx[::-1]
                
                self.set_sx_index(event, s1_idx, s2_idx)
                self.set_event_properties(event, largest_s1s, largest_s2s, peaks)

                # Loop over S1s and S2s and over main / alt.
                for s_i, largest_s_i in enumerate([largest_s1s, largest_s2s], 1):
                    # Largest index 0 -> main sx, 1 -> alt sx
                    for largest_index, main_or_alt in enumerate(['s', 'alt_s']):
                        peak_properties_to_save = [name for name, _, _ in self.peak_properties]
                        if s_i == 2:
                            peak_properties_to_save += ['x', 'y']
                            peak_properties_to_save += self.posrec_save
                        field_names = [f'{main_or_alt}{s_i}_{name}' for name in peak_properties_to_save]
                        self.copy_largest_peaks_into_event(event,
                                                        largest_s_i,
                                                        largest_index,
                                                        field_names,
                                                        peak_properties_to_save)

    @staticmethod
    @numba.njit
    def set_event_properties(result, largest_s1s, largest_s2s, peaks):
        """Get properties like drift time and area before main S2"""
        # Compute drift times only if we have a valid S1-S2 pair
        if len(largest_s1s) > 0 and len(largest_s2s) > 0:
            result['drift_time'] = largest_s2s[0]['center_time'] - largest_s1s[0]['center_time']

            # Correcting alt S1 and S2 based on BiPo 

            if len(largest_s1s) > 1:
                result['alt_s1_interaction_drift_time'] = largest_s2s[1]['center_time'] - largest_s1s[1]['center_time']
                result['alt_s1_delay'] = largest_s1s[1]['center_time'] - largest_s1s[0]['center_time']
            if len(largest_s2s) > 1:
                result['alt_s2_interaction_drift_time'] = largest_s2s[1]['center_time'] - largest_s1s[1]['center_time']
                result['alt_s2_delay'] = largest_s2s[1]['center_time'] - largest_s2s[0]['center_time']

        # areas before main S2
        if len(largest_s2s):
            peaks_before_ms2 = peaks[peaks['time'] < largest_s2s[0]['time']]
            result['area_before_main_s2'] = np.sum(peaks_before_ms2['area'])

            s2peaks_before_ms2 = peaks_before_ms2[peaks_before_ms2['type'] == 2]
            if len(s2peaks_before_ms2) == 0:
                result['large_s2_before_main_s2'] = 0
            else:
                result['large_s2_before_main_s2'] = np.max(s2peaks_before_ms2['area'])
        return result















@export
class EventPeaks(strax.Plugin):
    """
    Add event number for peaks and drift times of all s2 depending on the largest s1.
    Link - https://xe1t-wiki.lngs.infn.it/doku.php?id=weiss:analysis:ms_plugin
    """
    __version__ = '0.0.1'
    depends_on = ('event_basics', 'peak_basics', 'peak_positions')
    provides = 'peak_per_event'
    data_kind = 'peaks'
    save_when = strax.SaveWhen.TARGET

    def infer_dtype(self):
        dtype = strax.time_fields + [
            ('drift_time', np.float32, 'Drift time between main S1 and S2 in ns'),
            ('event_number', np.int64, 'Event number in this dataset'),
        ]
        return dtype

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        split_peaks_ind = strax.fully_contained_in(peaks, events)
        result = np.zeros(len(peaks), self.dtype)
        result.fill(np.nan)

        # Assign peaks features to main S1 and main S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            result['drift_time'][split_peaks_ind==event_i] = sp['center_time'] - event['s1_center_time']
        result['event_number'] = split_peaks_ind
        result['drift_time'][peaks['type'] != 2] = np.nan
        result['time'] = peaks['time']
        result['endtime'] = strax.endtime(peaks)
        return result



import numpy as np
import strax
import straxen
from straxen.common import get_resource, rotate_perp_wires
DEFAULT_POSREC_ALGO = 'mlp'

@export
class CorrectedAreas(strax.Plugin):
    """
    Plugin which applies light collection efficiency maps and electron
    life time to the data.
    Computes the cS1/cS2 for the main/alternative S1/S2 as well as the
    corrected life time.
    Note:
        Please be aware that for both, the main and alternative S1, the
        area is corrected according to the xy-position of the main S2.
        There are now 3 components of cS2s: cs2_top, cS2_bottom and cs2.
        cs2_top and cs2_bottom are corrected by the corresponding maps,
        and cs2 is the sum of the two.
    """
    __version__ = '0.3.0'

    depends_on = ['event_basics', 'event_positions']

    # Descriptor configs
    elife = straxen.URLConfig(
        default='cmt://elife?version=ONLINE&run_id=plugin.run_id',
        help='electron lifetime in [ns]')

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO,
        help="default reconstruction algorithm that provides (x,y)"
    )
    s1_xyz_map = straxen.URLConfig(
        default='itp_map://resource://cmt://format://'
                's1_xyz_map_{algo}?version=ONLINE&run_id=plugin.run_id'
                '&fmt=json&algo=plugin.default_reconstruction_algorithm',
        cache=True)
    s2_xy_map = straxen.URLConfig(
        default='itp_map://resource://cmt://format://'
                's2_xy_map_{algo}?version=ONLINE&run_id=plugin.run_id'
                '&fmt=json&algo=plugin.default_reconstruction_algorithm',
        cache=True)

    # average SE gain for a given time period. default to the value of this run in ONLINE model
    # thus, by default, there will be no time-dependent correction according to se gain
    avg_se_gain = straxen.URLConfig(
        default='cmt://avg_se_gain?version=ONLINE&run_id=plugin.run_id',
        help='Nominal single electron (SE) gain in PE / electron extracted. '
             'Data will be corrected to this value')

    # se gain for this run, allowing for using CMT. default to online
    se_gain = straxen.URLConfig(
        default='cmt://se_gain?version=ONLINE&run_id=plugin.run_id',
        help='Actual SE gain for a given run (allows for time dependence)')

    # relative extraction efficiency which can change with time and modeled by CMT.
    rel_extraction_eff = straxen.URLConfig(
        default='cmt://rel_extraction_eff?version=ONLINE&run_id=plugin.run_id',
        help='Relative extraction efficiency for this run (allows for time dependence)')

    # relative light yield
    # defaults to no correction
    rel_light_yield = straxen.URLConfig(
        default='cmt://relative_light_yield?version=ONLINE&run_id=plugin.run_id',
        help='Relative light yield (allows for time dependence)'
    )

    region_linear = straxen.URLConfig(
        default=28,
        help='linear cut (cm) for ab region, check out the note https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:sr0_2_region_se_correction'
    )

    region_circular = straxen.URLConfig(
        default=60,
        help='circular cut (cm) for ab region, check out the note https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:sr0_2_region_se_correction'
    )

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields

        for peak_type, peak_name in zip(['', 'alt_'], ['main', 'alternate']):
            dtype += [(f'{peak_type}cs1', np.float32, f'Corrected area of {peak_name} S1 [PE]'),
                      (f'{peak_type}cs1_wo_timecorr', np.float32,
                       f'Corrected area of {peak_name} S1 [PE] before time-dep LY correction'),
                      (f'{peak_type}cs2_wo_elifecorr', np.float32,
                       f'Corrected area of {peak_name} S2 before elife correction '
                       f'(s2 xy correction + SEG/EE correction applied) [PE]'),
                      (f'{peak_type}cs2_wo_timecorr', np.float32,
                       f'Corrected area of {peak_name} S2 before SEG/EE and elife corrections'
                       f'(s2 xy correction applied) [PE]'),
                      (f'{peak_type}cs2_area_fraction_top', np.float32,
                       f'Fraction of area seen by the top PMT array for corrected {peak_name} S2'),
                      (f'{peak_type}cs2_bottom', np.float32,
                       f'Corrected area of {peak_name} S2 in the bottom PMT array [PE]'),
                      (f'{peak_type}cs2', np.float32, f'Corrected area of {peak_name} S2 [PE]'), ]
        return dtype

    def ab_region(self, x, y):
        new_x, new_y = rotate_perp_wires(x, y)
        cond = new_x < self.region_linear
        cond &= new_x > -self.region_linear
        cond &= new_x ** 2 + new_y ** 2 < self.region_circular ** 2
        return cond

    def cd_region(self, x, y):
        return ~self.ab_region(x, y)
    
    def s2_map_names(self):
        
        # S2 top and bottom are corrected separately, and cS2 total is the sum of the two
        # figure out the map name
        if len(self.s2_xy_map.map_names) > 1:
            s2_top_map_name = "map_top"
            s2_bottom_map_name = "map_bottom"
        else:
            s2_top_map_name = "map"
            s2_bottom_map_name = "map"
        
        return s2_top_map_name, s2_bottom_map_name
    
    def seg_ee_correction_preparation(self):
        """Get single electron gain and extraction efficiency options"""
        self.regions = {'ab': self.ab_region, 'cd': self.cd_region}
        
        # setup SEG and EE corrections
        # if they are dicts, we just leave them as is
        # if they are not, we assume they are floats and
        # create a dict with the same correction in each region
        if isinstance(self.se_gain, dict):
            seg = self.se_gain
        else:
            seg = {key: self.se_gain for key in self.regions}

        if isinstance(self.avg_se_gain, dict):
            avg_seg = self.avg_se_gain
        else:
            avg_seg = {key: self.avg_se_gain for key in self.regions}

        if isinstance(self.rel_extraction_eff, dict):
            ee = self.rel_extraction_eff
        else:
            ee = {key: self.rel_extraction_eff for key in self.regions}
        
        return seg, avg_seg, ee
    
    def compute(self, events):
        result = np.zeros(len(events), self.dtype)
        result['time'] = events['time']
        result['endtime'] = events['endtime']

        # S1 corrections depend on the actual corrected event position.
        # We use this also for the alternate S1; for e.g. Kr this is
        # fine as the S1 correction varies slowly.
        event_positions = np.vstack([events['x'], events['y'], events['z']]).T

        for peak_type in ["", "alt_"]:
            result[f"{peak_type}cs1_wo_timecorr"] = events[f'{peak_type}s1_area'] / self.s1_xyz_map(event_positions)
            result[f"{peak_type}cs1"] = result[f"{peak_type}cs1_wo_timecorr"] / self.rel_light_yield

        # s2 corrections
         # s2 corrections
        s2_top_map_name, s2_bottom_map_name = self.s2_map_names()

        seg, avg_seg, ee = self.seg_ee_correction_preparation()

        # now can start doing corrections
        for peak_type in ["", "alt_"]:
            # S2(x,y) corrections use the observed S2 positions
            s2_positions = np.vstack([events[f'{peak_type}s2_x'], events[f'{peak_type}s2_y']]).T

            # corrected s2 with s2 xy map only, i.e. no elife correction
            # this is for s2-only events which don't have drift time info

            cs2_top_xycorr = (events[f'{peak_type}s2_area']
                              * events[f'{peak_type}s2_area_fraction_top']
                              / self.s2_xy_map(s2_positions, map_name=s2_top_map_name))
            cs2_bottom_xycorr = (events[f'{peak_type}s2_area']
                                 * (1 - events[f'{peak_type}s2_area_fraction_top'])
                                 / self.s2_xy_map(s2_positions, map_name=s2_bottom_map_name))

            # For electron lifetime corrections to the S2s,
            # use drift time computed using the main S1.
            el_string = peak_type + "s2_interaction_" if peak_type == "alt_" else peak_type
            elife_correction = np.exp(events[f'{el_string}drift_time'] / self.elife)
            result[f"{peak_type}cs2_wo_timecorr"] = (cs2_top_xycorr + cs2_bottom_xycorr) * elife_correction

            for partition, func in self.regions.items():
                # partitioned SE and EE
                partition_mask = func(events[f'{peak_type}s2_x'], events[f'{peak_type}s2_y'])

                # Correct for SEgain and extraction efficiency
                seg_ee_corr = seg[partition] / avg_seg[partition] * ee[partition]

                # note that these are already masked!
                cs2_top_wo_elifecorr = cs2_top_xycorr[partition_mask] / seg_ee_corr
                cs2_bottom_wo_elifecorr = cs2_bottom_xycorr[partition_mask] / seg_ee_corr

                result[f"{peak_type}cs2_wo_elifecorr"][partition_mask] = cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr

                # cs2aft doesn't need elife/time corrections as they cancel
                result[f"{peak_type}cs2_area_fraction_top"][partition_mask] = cs2_top_wo_elifecorr / \
                                                                              (
                                                                                      cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr)

                result[f"{peak_type}cs2"][partition_mask] = result[f"{peak_type}cs2_wo_elifecorr"][partition_mask] * \
                                                            elife_correction[partition_mask]
                result[f"{peak_type}cs2_bottom"][partition_mask] = cs2_bottom_wo_elifecorr * elife_correction[
                    partition_mask]

        return result


import strax
import numpy as np
import straxen

@export
class PeakCorrectedAreas(CorrectedAreas):
    """
    Pluging to apply corrections on peak level assuming that the main 
    S1 is the only physical S1. 
    """
    __version__ = '0.0.1'

    depends_on = ('peak_basics', 'peak_positions', 'peak_per_event')
    data_kind = 'peaks'
    provides = 'peak_corrections'

    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

    electron_drift_time_gate = straxen.URLConfig(
        default='cmt://'
                'electron_drift_time_gate'
                '?version=ONLINE&run_id=plugin.run_id',
        help='Electron drift time from the gate in ns',
        cache=True)

    def infer_dtype(self):
        dtype = strax.time_fields + [
            (('Corrected area of S2 before elife correction '
              '(s2 xy correction + SEG/EE correction applied) [PE]',
              'cs2_wo_elifecorr'), np.float32),
            (('Corrected area of S2 before SEG/EE and elife corrections '
              '(s2 xy correction applied) [PE]',
              'cs2_wo_timecorr'), np.float32),
            (('Fraction of area seen by the top PMT array for corrected S2',
              'cs2_area_fraction_top'), np.float32),
            (('Corrected area of S2 in the bottom PMT array [PE]',
              'cs2_bottom'), np.float32),
            (('Corrected area of S2 [PE]', 'cs2'), np.float32),
            (('Correction factor for the S1 area based on S2 position',
              's1_xyz_correction_factor'), np.float32),
            (('Relative light yield correction factor for the S1 area',
              's1_rel_light_yield_correction_factor'), np.float32),
            (('z position of the multiscatter peak',
              'z_obs_ms'), np.float32),
        ]
        return dtype

    def compute(self, peaks):
        result = np.zeros(len(peaks), self.dtype)
        result['time'] = peaks['time']
        result['endtime'] = peaks['endtime']

        # Get z position of the peak
        z_obs = -self.electron_drift_velocity * peaks['drift_time']
        z_obs = z_obs + self.electron_drift_velocity * self.electron_drift_time_gate
        result['z_obs_ms'] = z_obs

        # Get S1 correction factors
        peak_positions = np.vstack([peaks['x'], peaks['y'], z_obs]).T
        result['s1_xyz_correction_factor'] = 1 / self.s1_xyz_map(peak_positions)
        result['s1_rel_light_yield_correction_factor'] = 1 / self.rel_light_yield

        # s2 corrections
        s2_top_map_name, s2_bottom_map_name = self.s2_map_names()

        seg, avg_seg, ee = self.seg_ee_correction_preparation()

        # now can start doing corrections

        # S2(x,y) corrections use the observed S2 positions
        s2_positions = np.vstack([peaks['x'], peaks['y']]).T

        # corrected s2 with s2 xy map only, i.e. no elife correction
        # this is for s2-only events which don't have drift time info

        cs2_top_xycorr = (peaks['area']
                          * peaks['area_fraction_top']
                          / self.s2_xy_map(s2_positions, map_name=s2_top_map_name))
        cs2_bottom_xycorr = (peaks['area']
                             * (1 - peaks['area_fraction_top'])
                             / self.s2_xy_map(s2_positions, map_name=s2_bottom_map_name))

        # For electron lifetime corrections to the S2s,
        # use drift time computed using the main S1.

        elife_correction = np.exp(peaks['drift_time'] / self.elife)
        result['cs2_wo_timecorr'] = ((cs2_top_xycorr + cs2_bottom_xycorr) * elife_correction)

        for partition, func in self.regions.items():
            # partitioned SE and EE
            partition_mask = func(peaks['x'], peaks['y'])

            # Correct for SEgain and extraction efficiency
            seg_ee_corr = seg[partition] / avg_seg[partition] * ee[partition]

            # note that these are already masked!
            cs2_top_wo_elifecorr = cs2_top_xycorr[partition_mask] / seg_ee_corr
            cs2_bottom_wo_elifecorr = cs2_bottom_xycorr[partition_mask] / seg_ee_corr

            result['cs2_wo_elifecorr'][partition_mask] = cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr

            # cs2aft doesn't need elife/time corrections as they cancel
            result['cs2_area_fraction_top'][partition_mask] = cs2_top_wo_elifecorr / (
                        cs2_top_wo_elifecorr + cs2_bottom_wo_elifecorr)

            result['cs2'][partition_mask] = result['cs2_wo_elifecorr'][
                partition_mask] * elife_correction[partition_mask]
            result['cs2_bottom'][partition_mask] = cs2_bottom_wo_elifecorr * elife_correction[partition_mask]

        not_s2_mask = peaks['type'] != 2
        result['cs2_wo_timecorr'][not_s2_mask] = np.nan
        result['cs2_wo_elifecorr'][not_s2_mask] = np.nan
        result['cs2_area_fraction_top'][not_s2_mask] = np.nan
        result['cs2'][not_s2_mask] = np.nan
        result['z_obs_ms'][not_s2_mask] = np.nan
        result['cs2_bottom'][not_s2_mask] = np.nan
        result['s1_xyz_correction_factor'][not_s2_mask] = np.nan
        result['s1_rel_light_yield_correction_factor'][not_s2_mask] = np.nan
        return result



import strax
import straxen
import numpy as np
export, __all__ = strax.exporter()


@export
class EventInfoMS(strax.Plugin):
    """
    Plugin to collect multiple-scatter event observables
    """
    __version__ = '0.0.2'
    depends_on = (
        'event_info',
        'peak_basics', 'peak_per_event', 'peak_corrections', 'peak_positions')
    provides = 'event_ms_naive'
    save_when = strax.SaveWhen.TARGET

    # config options don't double cache things from the resource cache!
    g1 = straxen.URLConfig(
        default='bodega://g1?bodega_version=v2',
        help='S1 gain in PE / photons produced',
    )
    g2 = straxen.URLConfig(
        default='bodega://g2?bodega_version=v2',
        help='S2 gain in PE / electrons produced',
    )
    lxe_w = straxen.URLConfig(
        default=13.7e-3,
        help='LXe work function in quanta/keV'
    )
    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )
    max_drift_length = straxen.URLConfig(
        default=straxen.tpc_z, infer_type=False,
        help='Total length of the TPC from the bottom of gate to the '
             'top of cathode wires [cm]')

    ms_window_fac = straxen.URLConfig(
        default=1.01, type=(int, float),
        help='Max drift time window to look for peaks in multiple scatter events'
    )

    def infer_dtype(self):
        dtype = strax.time_fields + [
            (('Sum of S1 areas in event',
              's1_sum'), np.float32),
            (('Corrected S1 area based on average position of S2s in event',
              'cs1_multi'), np.float32),
            (('Corrected S1 area based on average position of S2s in event before time-dep LY correction',
              'cs1_multi_wo_timecorr'), np.float32),
            (('Sum of S2 areas in event',
              's2_sum'), np.float32),
            (('Sum of corrected S2 areas in event',
              'cs2_sum'), np.float32),
            (('Sum of corrected S2 areas in event S2 before elife correction',
              'cs2_wo_timecorr_sum'), np.float32),
            (('Sum of corrected S2 areas in event before SEG/EE and elife corrections',
              'cs2_wo_elifecorr_sum'), np.float32),
            (('Average of S2 area fraction top in event',
              'cs2_area_fraction_top_avg'), np.float32),
            (('Sum of the energy estimates in event',
              'ces_sum'), np.float32),
            (('Sum of the charge estimates in event',
              'e_charge_sum'), np.float32),
            (('Average x position of S2s in event',
              'x_avg'), np.float32),
            (('Average y position of S2s in event',
              'y_avg'), np.float32),
            (('Average observed z position of energy deposits in event',
              'z_obs_avg'), np.float32),
            (('Number of S2s in event',
              'multiplicity'), np.int32),
        ]
        return dtype

    def setup(self):
        self.drift_time_max = int(self.max_drift_length / self.electron_drift_velocity)

    def cs1_to_e(self, x):
        return self.lxe_w * x / self.g1

    def cs2_to_e(self, x):
        return self.lxe_w * x / self.g2

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        result = np.zeros(len(events), self.infer_dtype())

        # Assign peaks features to main S1 and main S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            cond = (sp['type'] == 2) & (sp['drift_time'] > 0)
            cond &= (sp['drift_time'] < self.ms_window_fac * self.drift_time_max) & (sp['cs2'] > 0)
            result['s2_sum'][event_i] = np.nansum(sp[cond]['area'])
            result['cs2_sum'][event_i] = np.nansum(sp[cond]['cs2'])
            result['cs2_wo_timecorr_sum'][event_i] = np.nansum(sp[cond]['cs2_wo_timecorr'])
            result['cs2_wo_elifecorr_sum'][event_i] = np.nansum(sp[cond]['cs2_wo_elifecorr'])         
            result['s1_sum'][event_i] = np.nansum(sp[sp['type'] == 1]['area'])

            if np.sum(sp[cond]['cs2']) > 0: 
                result['cs1_multi_wo_timecorr'][event_i] = event['s1_area'] * np.average(
                    sp[cond]['s1_xyz_correction_factor'], weights=sp[cond]['cs2'])
                result['cs1_multi'][event_i] = result['cs1_multi_wo_timecorr'][event_i] * np.average(
                    sp[cond]['s1_rel_light_yield_correction_factor'], weights=sp[cond]['cs2'])
                result['x_avg'][event_i] = np.average(sp[cond]['x'], weights=sp[cond]['cs2'])
                result['y_avg'][event_i] = np.average(sp[cond]['y'], weights=sp[cond]['cs2'])
                result['z_obs_avg'][event_i] = np.average(sp[cond]['z_obs_ms'], weights=sp[cond]['cs2'])
                result['cs2_area_fraction_top_avg'][event_i] = np.average(
                    sp[cond]['cs2_area_fraction_top'], weights=sp[cond]['cs2'])   
                result['multiplicity'][event_i] = len(sp[cond]['area'])

        el = self.cs1_to_e(result['cs1_multi'])
        ec = self.cs2_to_e(result['cs2_sum'])
        result['ces_sum'] = el + ec
        result['e_charge_sum'] = ec
        result['time'] = events['time']
        result['endtime'] = strax.endtime(events)
        return result
