import sys
sys.path.append('analysis_modules')
import os
from optparse import OptionParser
import pickle

import numpy as np
import matplotlib.pyplot as plt

import odor_control_for_pulse_firmware as ocf
import fit_odor_dataset_with_gaussians as fodwg

def get_led_position_vector():
    led_position = np.array([.74745, -.01065, .05581])
    tip_position = np.array([.6340, .0137, 0.0])
    led_position_vector = led_position - tip_position
    return led_position_vector

#####################################################################
# Get trajectories
#####################################################################

def get_trajectories_with_odor(dataset, threshold=50, odor_stimulus='pulsing'):
    keys = []
    for key, trajec in dataset.trajecs.items():
        if np.max(trajec.odor) > threshold:
            if trajec.odor_stimulus == odor_stimulus:
                keys.append(key)
    return keys
        
def get_trajectory_with_max_odor_signal(dataset, odor_stimulus='pulsing'):
    max_odor = 0
    max_key = None
    for key, trajec in dataset.trajecs.items():
        if trajec.odor_stimulus == odor_stimulus:
            if np.max(trajec.odor) > max_odor:
                max_key = key
                max_odor = np.max(trajec.odor)
    if max_key is not None:
        return dataset.trajecs[max_key]
    else:
        return None
        
def get_keys_with_odor_before_post(config, dataset, threshold_odor=50, odor_stimulus='pulsing', threshold_distance_min=0.1, odor=True, post_behavior=None):
    keys = []
    for key, trajec in dataset.trajecs.items():
        add_key = True
        indices_in_odor = np.where(trajec.odor > threshold_odor)[0]
        index_at_max_odor = np.argmax(trajec.odor)
        position_at_max_odor = trajec.positions[index_at_max_odor]
        signed_distance_to_post = position_at_max_odor[0] - config.post_center[0]
                
        if trajec.odor_stimulus == odor_stimulus: pass
        else: add_key = False
        
        if np.max(trajec.distance_to_post) > 0.05: pass
        else: add_key = False
            
        odor_check = False
        if odor:
            if np.max(trajec.odor) > threshold_odor:
                odor_check = True
        else:
            if np.max(trajec.odor) < threshold_odor:
                odor_check = True
        
        if odor_check: pass
        else: add_key = False
            
        
        if trajec.distance_to_post_min < threshold_distance_min: pass
        else: add_key = False
        
            
        if post_behavior is not None:
            if post_behavior not in trajec.post_behavior:
                add_key = False
        
        if add_key:
            keys.append(key)
                                
                            
    return keys
    
def get_keys_with_odor_before_post_fictive(config, dataset, threshold_odor=50, threshold_distance=0.1, upwind_only=True, odor_stimulus='pulsing'):
    keys = []
    for key, trajec in dataset.trajecs.items():
        if trajec.odor_stimulus == odor_stimulus:
            if trajec.odor_fictive is not None:
                if np.max(trajec.odor_fictive) > threshold_odor:
                    index_at_max_odor = np.argmax(trajec.odor_fictive)
                    position_at_max_odor = trajec.positions[index_at_max_odor]
                    signed_distance_to_post = position_at_max_odor[0] - config.post_center[0]
                    
                    if signed_distance_to_post > threshold_distance:
                        
                        if upwind_only:                    
                            indices_in_odor = np.where(trajec.odor_fictive > threshold_odor)[0]
                            avg_x_fly_direction_in_odor = np.mean(np.diff(trajec.positions[indices_in_odor.tolist()][:,0]))
                            if avg_x_fly_direction_in_odor < 0:
                                keys.append(key)
                        else:
                            keys.append(key)
    return keys
    
    
def get_frames_after_odor(dataset, keys, frames_to_show_before_odor=25):
    frames = []
    for key in keys:
        trajec = dataset.trajecs[key]
        frame0 = np.argmax(trajec.odor)-frames_to_show_before_odor
        frame0 = np.max([0, frame0]) # make sure don't get negative frames!
        frames_for_trajec = np.arange(frame0, trajec.length-1)
        frames.append(frames_for_trajec)
    return frames
            
#####################################################################
# 
#####################################################################

        
def calc_odor_signal_for_trajectories(path, dataset, config=None, gm_file=None):

    if config is None:
        sys.path.append(path)
        import analysis_configuration
        config = analysis_configuration.Config()

    print 'fitting gaussian model'
    gm = get_gaussian_model(path, config)
    print
    print 'Parameters: '
    print gm.parameters

    print 'getting timestamps from odor signal file'
    timestamps_signal_on = get_timestamps_epoch_from_odor_control_signal_file(path, config)
    
    print 'calculating odor traces for trajectories'
    for key, trajec in dataset.trajecs.items():
        calc_odor_for_trajectory(config, trajec, timestamps_signal_on, gm)
        
    return gm
    
def get_timestamps_epoch_from_odor_control_signal_file(path, config):
    assert config.odor is True
    odor_control_signal_filename_with_path_list = []
    
    if config.odor_control_files is None:
        odor_control_path = os.path.join(path, config.odor_control_path)
        cmd = 'ls ' + odor_control_path
        ls = os.popen(cmd).read()
        all_filelist = ls.split('\n')
        try: all_filelist.remove('')
        except: pass

        for odor_control_signal_filename in all_filelist:
            odor_control_signal_filename_with_path = os.path.join(odor_control_path, odor_control_signal_filename)
            odor_control_signal_filename_with_path_list.append(odor_control_signal_filename_with_path)
    #######
    timestamps_signal_on = []
    
    for odor_control_signal_filename_with_path in odor_control_signal_filename_with_path_list:
        odor_file = open(odor_control_signal_filename_with_path, 'r')
        lines = odor_file.readlines()
        
        # first extract time_start
        time_start_line = lines[0]
        time_start_string = time_start_line.split(',')[1][0:-1]
        time_start = float(time_start_string)
        
        # make list of ranges when odor is on
        
        for l, line in enumerate(lines[1:]):
            line = line.strip()
            command = line.split(',')[0]
            if command == 'on':
                time_epoch_str = line.split(',')[2]
                if len(time_epoch_str.split('.')[1]) <= 2:
                    time_epoch = float(time_epoch_str)
                else:
                    time_epoch = float(time_epoch_str) + time_start
                timestamps_signal_on.append(time_epoch)
        
    return np.array(timestamps_signal_on)
    
def get_time_relative_to_last_odor_pulse(t, timestamps_signal_on, delay=12):
    # find pulse closest to about delay (12) seconds ago (most likely to have odor then)
    try:
        most_recent_pulse_index = np.argmin(np.abs(t- (timestamps_signal_on+12) )) # np.where((t- timestamps_signal_on)>0)[0][-1]
        t_relative_to_signal = t - timestamps_signal_on[most_recent_pulse_index]
    except:
        t_relative_to_signal = 0
    return t_relative_to_signal

def calc_odor_stimulus_for_trajectory(config, trajec):
    
    def inrange(val, testrange):
        if val > np.min(testrange) and val < np.max(testrange):
            return True
        else:
            return False
            
    for stimulus in config.odor_stimulus.keys():
        if inrange(trajec.timestamp_local_float, config.odor_stimulus[stimulus]):
            trajec.odor_stimulus = stimulus
            
    
def calc_odor_for_trajectory(config, trajec, timestamps_signal_on, gm):
    
    trajec.odor = np.zeros_like(trajec.speed)
    trajec.time_relative_to_last_pulse = np.zeros_like(trajec.speed)
    calc_odor_stimulus_for_trajectory(config, trajec)
    
    static_time = 12.
    gm_static = gm.get_gaussian_model_at_time_t(static_time)
    
    # set x axis to be huge
    gm_static.parameters['mean_0'] = 0
    gm_static.parameters['std_0'] = 10000000000000000
    
    if trajec.odor_stimulus == 'pulsing':
        for i, val in enumerate(trajec.odor):
            t = trajec.timestamp_epoch + trajec.time_fly[i]
            trajec.time_relative_to_last_pulse[i] = get_time_relative_to_last_odor_pulse(t, timestamps_signal_on)
            x,y,z = trajec.positions[i]
            trajec.odor[i] = gm.get_val([trajec.time_relative_to_last_pulse[i], [x,y,z]])
    elif trajec.odor_stimulus == 'none':
    
        if 0:
            # calculate fictive odor signal
            fictive_range = [-5, 5]
            fictive_pulse_time = np.random.ranf()*(fictive_range[1]-fictive_range[0])+fictive_range[0]
            for i, val in enumerate(trajec.odor):
                trajec.time_relative_to_last_pulse[i] = trajec.time_fly[i] - fictive_pulse_time
                x,y,z = trajec.positions[i]
                trajec.odor[i] = gm.get_val([trajec.time_relative_to_last_pulse[i], [x,y,z]])
        if 1:
            for i, val in enumerate(trajec.odor):
                t = 12
                trajec.time_relative_to_last_pulse[i] = static_time
                x,y,z = trajec.positions[i]
                trajec.odor[i] = gm_static.get_val([x,y,z])
                
    elif trajec.odor_stimulus == 'on':
        for i, val in enumerate(trajec.odor):
            t = 12
            trajec.time_relative_to_last_pulse[i] = static_time
            x,y,z = trajec.positions[i]
            trajec.odor[i] = gm_static.get_val([x,y,z])
            
            
def get_gaussian_model(path, config):

    if config.odor_gaussian_fit is not None:
        savetopath = os.path.join(path, config.odor_gaussian_fit)
        print 'will save to: ', savetopath
        try:
            fd = open(savetopath, 'r')
            gm = pickle.load(fd)
            fd.close()
            print 'loaded gm from file: ', config.odor_gaussian_fit
        except:
    
            odor_packet_data_path = config.odor_packet_data_path

            led_position_vector = get_led_position_vector()
            #led_position_vector = np.array([0,0,0])
            odor_dataset = ocf.get_odor_dataset_from_means(odor_packet_data_path, led_position_vector=led_position_vector)
            
            gm = fodwg.fit_gaussian3d_timevarying(odor_dataset)
            
            print 'saving gm to file...'
            fd = open(savetopath, 'w')
            pickle.dump(gm, fd)
            fd.close()
    
    return gm

def plot_gaussian_model_vs_raw_data(gm, filename, save=True):
    f = open(filename)
    odor_dataset = pickle.load(f)
    led_position_vector = get_led_position_vector()
    ocf.calc_mean_odor_trace(odor_dataset, led_position_vector=led_position_vector)      
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ocf.plot_mean_odor_trace(odor_dataset, ignore_traces=[1,2,3,4], ax=ax)
    
    mot = odor_dataset.mean_odor_trace
    timerange=[np.min(mot.timestamps), np.max(mot.timestamps)]
    timestamps, odor_vals = gm.get_time_trace_at_point(timerange, mot.position)
    
    ax.plot(timestamps, odor_vals, color='blue')
    
    path = os.path.dirname(filename)
    name = os.path.basename(filename) + '_gm_comparison.pdf'
    plotname = os.path.join(path, name)
    fig.savefig(plotname, format='pdf')
    
def plot_gaussian_model_vs_raw_data_for_all_traces(gm, path_to_odor_packets):
    cmd = 'ls ' + path_to_odor_packets
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    try: all_filelist.remove('')
    except: pass
    
    for filename in all_filelist:
        filename_with_path = os.path.join(path_to_odor_packets, filename)
        plot_gaussian_model_vs_raw_data(gm, filename_with_path)
        
        
def print_interesting_keys(config, dataset, threshold_odor=75, threshold_distance=0.01, max_distance=0.15, odor_stimulus='pulsing', upwind_only=True, keys=None):
    if keys is None:
        keys = get_keys_with_odor_before_post(config, dataset, threshold_odor=threshold_odor, threshold_distance=threshold_distance, odor_stimulus=odor_stimulus, upwind_only=upwind_only)
    min_distances = []
    time_spent_at_distance_threshold = []
    distances_at_max_odor = []
    mean_speed_before_odor = []
    mean_speed_in_odor = []
    mean_flight_speed_increase = []
    mean_speed_after_odor = []
    inter_saccade_interval_after_odor = []
    inter_saccade_interval_before_odor = []
    
    special_keys = []

    for key in keys:
        trajec = dataset.trajecs[key]
        frame_max_odor = np.argmax(trajec.odor)
        distance_at_max_odor = trajec.distance_to_post[frame_max_odor]
        if distance_at_max_odor < max_distance:
            if trajec.distance_to_post_min_index > frame_max_odor:
                frames_where_odor = np.where(trajec.odor > threshold_odor)[0]
                print key, ' -- max odor: ', np.max(trajec.odor), ' -- n frames in odor: ', len(frames_where_odor), ' -- behavior: ', trajec.post_behavior, ' -- min dist to post: ', np.min(trajec.distance_to_post), ' -- speed at min dist: ', trajec.speed[trajec.distance_to_post_min_index], ' -- localtime: ', trajec.timestamp_local_float
                
                min_distances.append(np.min(trajec.distance_to_post))
                distances_at_max_odor.append(distance_at_max_odor)
                
                frame0 = np.max([25, frames_where_odor[0]-100])
                if frame0 < frames_where_odor[0]-50:
                    mean_speed_before_odor.append(np.mean(trajec.speed[frame0:frames_where_odor[0]-50]))
                    mean_speed_in_odor.append(np.mean(trajec.speed[frames_where_odor]))
                    if frames_where_odor[-1] < trajec.distance_to_post_min_index:
                        mean_speed_after_odor.append(np.mean(trajec.speed[frames_where_odor[-1]:trajec.distance_to_post_min_index]))
                
                    mean_flight_speed_increase.append(mean_speed_in_odor[-1] - mean_speed_before_odor[-1])
                
                if len(frames_where_odor) > 10:
                    special_keys.append(key)
                
                # inter_saccade_interval_after_odor:
                inter_saccade_intervals_after = []
                inter_saccade_intervals_before = []
                for s, sacrange in enumerate(trajec.saccades):
                    if s == len(trajec.saccades)-1:
                        continue
                    if sacrange[-1] > frames_where_odor[-1]:
                        interval = trajec.saccades[s+1][0] - sacrange[-1] 
                        inter_saccade_intervals_after.append(interval)
                for s, sacrange in enumerate(trajec.saccades):
                    if s == 0:
                        continue
                    if sacrange[-1] < frames_where_odor[0]:
                        interval = sacrange[0] - trajec.saccades[s-1][-1]
                        inter_saccade_intervals_before.append(interval)
                        
                if len(inter_saccade_intervals_after) > 0:
                    inter_saccade_interval_after_odor.append(np.mean(inter_saccade_intervals_after))
                if len(inter_saccade_intervals_before) > 0:
                    inter_saccade_interval_before_odor.append(np.mean(inter_saccade_intervals_before))
                
                # frames where fly closer than X to post
                dist_threshold = 0.03
                frames_close_to_post = np.where(trajec.distance_to_post < dist_threshold)[0]
                frames_close_to_post_after_odor = frames_close_to_post #> frame_max_odor
                time_spent_at_distance_threshold.append(len(frames_close_to_post_after_odor))
                
            
    print
    print 'mean min distance: ', np.mean(min_distances), ' +/- ', np.std(min_distances)
    print 'mean distance at odor: ', np.mean(distances_at_max_odor), ' +/- ', np.std(distances_at_max_odor)
    print 'mean frames spend close to post: ', np.mean(time_spent_at_distance_threshold), ' +/- ', np.std(time_spent_at_distance_threshold)
    print 'mean speed before odor: ', np.mean(mean_speed_before_odor), ' +/- ', np.std(mean_speed_before_odor)
    print 'mean speed in odor: ', np.mean(mean_speed_in_odor), ' +/- ', np.std(mean_speed_in_odor)
    print 'mean speed after odor: ', np.mean(mean_speed_after_odor), ' +/- ', np.std(mean_speed_after_odor)
    print 'mean flight speed increase: ', np.mean(mean_flight_speed_increase), ' +/- ', np.std(mean_flight_speed_increase)
    print 'mean inter saccade interval before odor: ', np.mean(inter_saccade_interval_before_odor), ' +/- ', np.std(inter_saccade_interval_before_odor)
    print 'mean inter saccade interval after odor: ', np.mean(inter_saccade_interval_after_odor), ' +/- ', np.std(inter_saccade_interval_after_odor)
    
    print 'special keys: ', special_keys
    
    return mean_flight_speed_increase
    
    
        
        
def min_distance_to_post_after_odor(config, dataset, threshold_odor=50, threshold_distance=0.01, max_distance = 0.15, odor_stimulus='pulsing', upwind_only=True):
    keys = get_keys_with_odor_before_post(config, dataset, threshold_odor=threshold_odor, threshold_distance=threshold_distance, odor_stimulus=odor_stimulus, upwind_only=upwind_only)
    
    distances = []
    for key in keys:
        trajec = dataset.trajecs[key]
        frame_max_odor = np.argmax(trajec.odor)
        distance_at_max_odor = trajec.distance_to_post[frame_max_odor]
        if distance_at_max_odor < max_distance:
            frames_after_odor = np.arange(frame_max_odor, trajec.length).tolist()
            min_distance_to_post = np.min(trajec.distance_to_post[frames_after_odor])
            distances.append(min_distance_to_post)
            print key, distance_at_max_odor, min_distance_to_post, trajec.post_behavior
    print 'mean distance: ', np.mean(distances)
    
    return np.array(distances)
    
'''
def min_distance_to_post_after_odor_fictive(config, dataset, threshold_odor=50, threshold_distance=0.01, max_distance = 0.15, odor_stimulus='pulsing', upwind_only=True):
    keys = get_keys_with_odor_before_post_fictive(config, dataset, threshold_odor=threshold_odor, threshold_distance=threshold_distance, odor_stimulus=odor_stimulus, upwind_only=upwind_only)
    print keys
    
    distances = []
    for key in keys:
        trajec = dataset.trajecs[key]
        if trajec.odor_fictive is not None:
            frame_max_odor = np.argmax(trajec.odor_fictive)
            distance_at_max_odor = trajec.distance_to_post[frame_max_odor]
            print distance_at_max_odor
            if distance_at_max_odor < max_distance:
                frames_after_odor = np.arange(frame_max_odor, trajec.length).tolist()
                min_distance_to_post = np.min(trajec.distance_to_post[frames_after_odor])
                distances.append(min_distance_to_post)
                print key, distance_at_max_odor, min_distance_to_post, trajec.post_behavior
    print 'mean distance: ', np.mean(distances)
    
    return np.array(distances)
'''
        
    
if __name__ == '__main__':
    pass 
    
