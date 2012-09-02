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
        
def get_keys_with_odor_before_post(config, dataset, threshold_odor=50, threshold_distance=0.1, upwind_only=True, odor_stimulus='pulsing'):
    keys = []
    for key, trajec in dataset.trajecs.items():
        if trajec.odor_stimulus == odor_stimulus:
            if np.max(trajec.odor) > threshold_odor:
                
                index_at_max_odor = np.argmax(trajec.odor)
                position_at_max_odor = trajec.positions[index_at_max_odor]
                signed_distance_to_post = position_at_max_odor[0] - config.post_center[0]
                
                if signed_distance_to_post > threshold_distance:
                    
                    if upwind_only:                    
                        indices_in_odor = np.where(trajec.odor > threshold_odor)[0]
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
    
    if config.odor_control_files is None:
        odor_control_path = os.path.join(path, config.odor_control_path)
        cmd = 'ls ' + odor_control_path
        ls = os.popen(cmd).read()
        all_filelist = ls.split('\n')
        try: all_filelist.remove('')
        except: pass

        odor_control_signal_filename = all_filelist[0]
        odor_control_signal_filename_with_path = os.path.join(odor_control_path, odor_control_signal_filename)
            
    odor_file = open(odor_control_signal_filename_with_path, 'r')
    lines = odor_file.readlines()
    
    # first extract time_start
    time_start_line = lines[0]
    time_start_string = time_start_line.split(',')[1][0:-1]
    time_start = float(time_start_string)
    
    # make list of ranges when odor is on
    timestamps_signal_on = []
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
    
def get_time_relative_to_last_odor_pulse(t, timestamps_signal_on):
    try:
        most_recent_pulse_index = np.where((t- timestamps_signal_on)>0)[0][-1]
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
    
    if trajec.odor_stimulus == 'pulsing':
        for i, val in enumerate(trajec.odor):
            t = trajec.timestamp_epoch + trajec.time_fly[i]
            trajec.time_relative_to_last_pulse[i] = get_time_relative_to_last_odor_pulse(t, timestamps_signal_on)
            x,y,z = trajec.positions[i]
            trajec.odor[i] = gm.get_val([trajec.time_relative_to_last_pulse[i], [x,y,z]])
    elif trajec.odor_stimulus == 'none':
        # calculate fictive odor signal
        for i, val in enumerate(trajec.odor):
            fictive_range = [-10, 0]
            fictive_pulse_time = np.random.ranf()*(fictive_range[1]-fictive_range[0])+fictive_range[0]
            trajec.time_relative_to_last_pulse[i] = trajec.time_fly[i] - fictive_pulse_time
            x,y,z = trajec.positions[i]
            trajec.odor[i] = gm.get_val([trajec.time_relative_to_last_pulse[i], [x,y,z]])
            
            
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

def plot_gaussian_model_vs_raw_data(gm, filename):
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
    
def plot_gaussian_model_vs_raw_data_for_all_traces(gm, path_to_odor_packets):
    cmd = 'ls ' + path_to_odor_packets
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    try: all_filelist.remove('')
    except: pass
    
    for filename in all_filelist:
        filename_with_path = os.path.join(path_to_odor_packets, filename)
        plot_gaussian_model_vs_raw_data(gm, filename_with_path)
    
if __name__ == '__main__':
    pass 
    
