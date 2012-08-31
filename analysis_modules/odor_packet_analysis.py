import sys
sys.path.append('analysis_modules')
import os
from optparse import OptionParser
import pickle

import numpy as np

import odor_control_for_pulse_firmware as ocf
import fit_odor_dataset_with_gaussians as fodwg

def get_trajectories_with_odor(dataset, threshold=50):
    keys = []
    for key, trajec in dataset.trajecs.items():
        if np.max(trajec.odor) > threshold:
            keys.append(key)
    return keys
        
    

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
        calc_odor_for_trajectory(trajec, timestamps_signal_on, gm)
        
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
    
def calc_odor_for_trajectory(trajec, timestamps_signal_on, gm):
    
    trajec.odor = np.zeros_like(trajec.speed)
    for val, i in enumerate(trajec.odor):
        t = trajec.timestamp_epoch + trajec.time_fly[i]
        t_relative_to_last_pulse = get_time_relative_to_last_odor_pulse(t, timestamps_signal_on)
        x,y,z = trajec.positions[i]
        trajec.odor[i] = gm.get_val([t_relative_to_last_pulse, [x,y,z]])
        
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

            led_position = np.array([.74745, -.01065, .05581])
            tip_position = np.array([.6340, .0137, 0.0])
            led_position_vector = led_position - tip_position
            #led_position_vector = np.array([0,0,0])
            odor_dataset = ocf.get_odor_dataset_from_means(odor_packet_data_path, led_position_vector=led_position_vector)
            
            gm = fodwg.fit_gaussian3d_timevarying(odor_dataset)
            
            print 'saving gm to file...'
            fd = open(savetopath, 'w')
            pickle.dump(gm, fd)
            fd.close()
    
    return gm


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to empty data folder, where you have a configuration file")
    (options, args) = parser.parse_args()
    
    path = options.path
    sys.path.append(path)
    import analysis_configuration
    config = analysis_configuration.Config()
    
    main(path, config)
