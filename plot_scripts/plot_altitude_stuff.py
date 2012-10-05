#!/usr/bin/env python
import sys, os, imp
sys.path.append('../')
sys.path.append('../analysis_modules')
from optparse import OptionParser
import flydra_analysis_tools as fat
import fly_plot_lib
fly_plot_lib.set_params.pdf()
import fly_plot_lib.plot as fpl
fad = fat.flydra_analysis_dataset
dac = fat.dataset_analysis_core
fap = fat.flydra_analysis_plot
tac = fat.trajectory_analysis_core

from flydra_analysis_tools import floris_math
from flydra_analysis_tools import kalman_math

import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

def in_range(val, minmax):
    if val > minmax[0] and val < minmax[1]:
        return True
    else:
        return False
        
    
def calc_acceleration(velocities):
    ## kalman
    
    data = velocities
    ss = 6 # state size
    os = 3 # observation size
    F = np.array([   [1,0,0,1,0,0], # process update
                     [0,1,0,0,1,0],
                     [0,0,1,0,0,1],
                     [0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1]],                  
                    dtype=np.float)
    H = np.array([   [1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]], # observation matrix
                    dtype=np.float)
    Q = 0.01*np.eye(ss) # process noise
    
    R = 1*np.eye(os) # observation noise
    
    initx = np.array([velocities[0,0], velocities[0,1], velocities[0,2], 0, 0, 0], dtype=np.float)
    initv = 0*np.eye(ss)
    xsmooth,Vsmooth = kalman_math.kalman_smoother(data, F, H, Q, R, initx, initv, plot=False)

    accel_smooth = xsmooth[:,3:]*100.
    
    return accel_smooth
        
        
def get_altitude_data(dataset, config, keys, odor_timerange=None, threshold_odor=10):
    altitudes = []
    #altitude_vels = []
    for key in keys:
        trajec = dataset.trajecs[key]
        #accel_smooth = calc_acceleration(trajec.velocities)[:,2]
        frames_where_odor = np.where(trajec.odor > threshold_odor)[0]
        saccade_frames = [item for sublist in trajec.saccades for item in sublist]
        frames_to_use = []
        for f, h in enumerate(trajec.heading_smooth):
            use_frame = True
            
            if odor_timerange is not None:
                tmp = f - np.array(frames_where_odor)
                try:
                    time_since_odor = tmp[np.where(tmp>0)[0][-1]]/trajec.fps
                except:
                    time_since_odor = 10000
                if not in_range(time_since_odor, odor_timerange):
                    use_frame = False
            
            if f in saccade_frames:
                use_frame = False
            if np.abs(trajec.positions[f,2]) > 0.1: # skip low and high points
                use_frame = False
            if use_frame:
                frames_to_use.append(f)
                
            if use_frame:
                altitudes.append(np.sum(trajec.velocities[f:f+20,2]))
            
        #altitudes.extend(trajec.positions[frames_to_use,2].tolist())
        #altitude_vels.extend(accel_smooth[frames_to_use].tolist())
        
    altitudes = np.array(altitudes)
    #altitude_vels = np.array(altitude_vels)
    
    return altitudes
    
    
    
    

# get altitude velocities for flies that are below odor, and recently experienced it
def get_altitude_stuff(config, dataset, visual_stimulus='none'):
    threshold_odor = 10
    
    keys_odor = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['on', visual_stimulus])
    keys_odor_true = []
    keys_odor_false = []
    for key in keys_odor:
        trajec = dataset.trajecs[key]
        if np.max(trajec.odor) > threshold_odor:
            keys_odor_true.append(key)
        else:
            keys_odor_false.append(key)
    #key_sets.setdefault('odor_true', keys_odor_true)
    #key_sets.setdefault('odor_false', keys_odor_false)
    
    
    altitudes = get_altitude_data(dataset, config, keys_odor_true, odor_timerange=[.5,1], threshold_odor=10)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(altitudes, altitude_vels, '.', markersize=.2)
    
    savename = 'altitudes.pdf'
    
    path = config.path
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    fig_name_with_path = os.path.join(save_figure_path, savename)

    print 'SAVING TO: ', fig_name_with_path
    fig.savefig(fig_name_with_path, format='pdf')
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to data folder, where you have a configuration file")
    (options, args) = parser.parse_args()
    
    path = options.path    
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
    dataset = fad.load(culled_dataset_filename)
        
        
    get_altitude_stuff(config, dataset)
        
    
    
    
    
        
    
