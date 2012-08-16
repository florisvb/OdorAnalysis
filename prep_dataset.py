#!/usr/bin/env python
import sys, os
from optparse import OptionParser

import flydra_analysis_tools as fat
fad = fat.flydra_analysis_dataset
dac = fat.dataset_analysis_core
tac = fat.trajectory_analysis_core

import numpy as np

import analysis_modules.odor_control_analysis as oca

####################### analysis functions ###########################
def culling_function(raw_dataset):
    culled_dataset = dac.cull_dataset_min_frames(raw_dataset, min_length_frames=50, reset=True)
    culled_dataset = dac.cull_dataset_cartesian_volume(culled_dataset, [-1,1], [-.3,.3], [-.3,.3], reset=True)
    culled_dataset = dac.cull_dataset_flight_envelope(culled_dataset, x_range=[-.1,.9], y_range=[-.12,.12], z_range=[-.1,.1], reset=True)
    culled_dataset = dac.cull_dataset_min_speed(culled_dataset, min_speed=0.01, reset=True)
    return culled_dataset


def prep_data(culled_dataset, path, config):
    # stuff like calculating angular velocity, saccades etc.
    
    for key, trajec in culled_dataset.trajecs.items():
        trajec.time_fly = np.linspace(0,trajec.length/trajec.fps,trajec.length, endpoint=True) 
        
    culled_dataset.info = config.info
    fad.iterate_calc_function(culled_dataset, tac.calc_local_timestamps_from_strings) # calculate local timestamps
    set_odor_stimulus(culled_dataset, config)
    
    if config.odor is True:
        oca.calc_odor_vs_no_odor_simple(culled_dataset, get_odor_control_filename(path, config))
    else:
        fad.set_attribute_for_trajecs(culled_dataset, 'in_odor', False)
        
    return    
    
def get_odor_control_filename(path, config):
    
    odor_control_path = os.path.join(path, config.odor_control_path)
    if 1: # default to find a file in the path
        try:
            cmd = 'ls ' + odor_control_path
            ls = os.popen(cmd).read()
            all_filelist = ls.split('\n')
        except:
            all_filelist = None
            print 'Could not find odor control path - might need to make it first'
        
        if all_filelist is not None:
            try:
                all_filelist.remove('')
            except:
                pass
            if len(all_filelist) == 1:
                odor_control_filename = all_filelist[0]
            else:
                print 'could not find unique odor control file in path'
            
    if 0: # hard code the odor control file
        odor_control_filename = config.odor_control_filename
        
    odor_control_filename = os.path.join(path, config.odor_control_path, odor_control_filename)
        
    return odor_control_filename
    

def set_odor_stimulus(dataset, config):

    def in_range(val, minmax):
        if val > minmax[0] and val < minmax[1]:
            return True
        else:
            return False
    
    for key, trajec in dataset.trajecs.items():
        trajec.odor_stimulus = 'none' # default
        for stim_name, minmax in config.odor_stimulus.items():
            if in_range(trajec.timestamp_local_float, minmax):
                trajec.odor_stimulus = stim_name
            
        
        
def main(path, config):
    
    # path stuff
    culled_dataset_name = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name)
    raw_dataset_name = os.path.join(path, config.raw_datasets_path, config.raw_dataset_name)
    
    print 
    print 'Culling and Preparing Data'
    
    try:
        culled_dataset = fad.load(culled_dataset_name)
        print 'Loaded culled dataset'
    except:
        try:
            raw_dataset = fad.load(raw_dataset_name)
            print 'Loaded raw dataset'
        except:
            print 'Cannot find dataset, run save_h5_to_dataset.py first'
                
        culled_dataset = culling_function(raw_dataset) 
        print 'Preparing culled dataset'
        prep_data(culled_dataset, path, config)
        fad.save(culled_dataset, culled_dataset_name)
        print 'Saved culled dataset'
        
    prep_data(culled_dataset, path, config)
    
    return culled_dataset
    
if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to empty data folder, where you have a configuration file")
    (options, args) = parser.parse_args()
    
    path = options.path    
    sys.path.append(path)
    import analysis_configuration
    config = analysis_configuration.Config()
    
    culled_dataset = main(path, config)
