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

def calc_heading(velocities):
    heading_norollover = floris_math.remove_angular_rollover(np.arctan2(velocities[:,1], velocities[:,0]), 3)
    ## kalman
    
    data = heading_norollover.reshape([len(heading_norollover),1])
    ss = 3 # state size
    os = 1 # observation size
    F = np.array([   [1,1,0], # process update
                     [0,1,1],
                     [0,0,1]],
                    dtype=np.float)
    H = np.array([   [1,0,0]], # observation matrix
                    dtype=np.float)
    Q = np.eye(ss) # process noise
    Q[0,0] = .01
    Q[1,1] = .01
    Q[2,2] = .01
    R = 1*np.eye(os) # observation noise
    
    initx = np.array([data[0,0], data[1,0]-data[0,0], 0], dtype=np.float)
    initv = 0*np.eye(ss)
    xsmooth,Vsmooth = kalman_math.kalman_smoother(data, F, H, Q, R, initx, initv, plot=False)

    heading_norollover_smooth = xsmooth[:,0]
    heading_smooth_diff = xsmooth[:,1]*100.
    
    heading = floris_math.fix_angular_rollover(heading_norollover)
    heading_smooth = floris_math.fix_angular_rollover(heading_norollover_smooth)
    
    if 0:
        flipm = np.where(heading_smooth < 0)[0]
        flipp = np.where(heading_smooth > 0)[0]
        
        heading_smooth[flipm] += np.pi
        heading_smooth[flipp] -= np.pi
        
    return heading_smooth

def in_range(val, minmax):
    if val > minmax[0] and val < minmax[1]:
        return True
    else:
        return False

def get_headings(dataset, config, keys, odor_timerange=None, threshold_odor=10, velocity_adjustment=0):
    

    headings = []
    for key in keys:
        trajec = dataset.trajecs[key]
        
        if velocity_adjustment != 0:
            velocities_adj = copy.copy(trajec.velocities)
            velocities_adj[:,0] += velocity_adjustment
            heading_adj = calc_heading(velocities_adj)
        
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
            
        if velocity_adjustment != 0:
            headings.extend(heading_adj[frames_to_use].tolist())
        else:
            headings.extend(trajec.heading_smooth[frames_to_use].tolist())
            
        
        #headings.extend(trajec.heading_smooth[frames_to_use].tolist())

    headings = np.array(headings)
    
    headings_less_than_zero = np.where(headings < 0)[0]
    headings_greater_than_zero = np.where(headings > 0)[0]
    
    headings[headings_less_than_zero] = headings[headings_less_than_zero] + np.pi
    headings[headings_greater_than_zero] = headings[headings_greater_than_zero] - np.pi
    
    return headings    
    
    
def get_altitude_headings(dataset, config, keys, odor_timerange=None, threshold_odor=10):
    

    headings = []
    for key in keys:
        trajec = dataset.trajecs[key]
        
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
            if np.abs(trajec.positions[f,2]) > 0.15: # skip low and high points
                use_frame = False
            if use_frame:
                frames_to_use.append(f)

        if 0:            
            altitude_speeds = np.vstack((trajec.speed_xy, trajec.velocities[:,2])).T
            altitude_headings = calc_heading(altitude_speeds)
            headings.extend(altitude_headings[frames_to_use].tolist())
        else:
            headings.extend((trajec.velocities[frames_to_use,2]/np.abs(trajec.velocities[frames_to_use,0])).tolist())
        
        #headings.extend(trajec.heading_smooth[frames_to_use].tolist())

    headings = np.array(headings)
    
    return headings    
    

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
                altitudes.append(np.sum(trajec.velocities[f:f+30,2])/20.)
            
        #altitudes.extend(trajec.positions[frames_to_use,2].tolist())
        #altitude_vels.extend(accel_smooth[frames_to_use].tolist())
        
    altitudes = np.array(altitudes)
    #altitude_vels = np.array(altitude_vels)
    
    return altitudes
    

def simple(dataset, config, visual_stimulus='none'):
    threshold_odor = 10
    key_sets = {}

    keys_odor = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['on', visual_stimulus])
    keys_odor_true = []
    keys_odor_false = []
    for key in keys_odor:
        trajec = dataset.trajecs[key]
        if np.max(trajec.odor) > threshold_odor:
            keys_odor_true.append(key)
        else:
            keys_odor_false.append(key)
    key_sets.setdefault('odor_true', keys_odor_true)
    if 0:
        key_sets.setdefault('odor_false', keys_odor_false)
            
    if 0:
        keys_noodor = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['afterodor', visual_stimulus])
        keys_noodor_true = []
        keys_noodor_false = []
        for key in keys_noodor:
            trajec = dataset.trajecs[key]
            if np.max(trajec.odor) > threshold_odor:
                keys_noodor_true.append(key)
            else:
                keys_noodor_false.append(key)
        key_sets.setdefault('noodor_true', keys_noodor_true)
        key_sets.setdefault('noodor_false', keys_noodor_false)
    else:
        keys_noodor = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['afterodor', visual_stimulus])
        key_sets.setdefault('noodor', keys_noodor)
        
    
    data = {}
    if 0:
        for key_set, keys in key_sets.items():
            heading = get_headings(dataset, config, keys)
            data.setdefault(key_set, heading)
            print key_set, len(keys)
        
    
    #colors = {'odor_true': 'red', 'odor_false': 'purple', 'noodor_true': 'black', 'noodor_false': 'blue'}
    colors = {'odor_true': 'red', 'noodor': 'black'}
        
    print 'colors: ', colors
    print 'data: ', data.keys()
    make_histograms(config, data, colors)
    
    

def visual_motion(dataset, config, dataset_control, config_control):

    print 'data: ', config.path
    print 'control: ', config_control.path
    
    threshold_odor = 10
    key_sets = {}
    data = {}
    
    keys_odor = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['on', 'upwind'])
    keys_odor_true = []
    keys_odor_false = []
    for key in keys_odor:
        trajec = dataset.trajecs[key]
        if np.max(trajec.odor) > threshold_odor:
            keys_odor_true.append(key)
        else:
            keys_odor_false.append(key)
    key_set = 'upwind'
    key_sets.setdefault(key_set, keys_odor_true)
    heading = get_headings(dataset, config, keys_odor_true, velocity_adjustment=0)
    data.setdefault(key_set, heading)
    print key_set, len(keys_odor_true)
            
    keys_odor = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['on', 'downwind'])
    keys_odor_true = []
    keys_odor_false = []
    for key in keys_odor:
        trajec = dataset.trajecs[key]
        if np.max(trajec.odor) > threshold_odor:
            keys_odor_true.append(key)
        else:
            keys_odor_false.append(key)
    key_set = 'downwind' 
    key_sets.setdefault(key_set, keys_odor_true)
    heading = get_headings(dataset, config, keys_odor_true, velocity_adjustment=0)
    data.setdefault(key_set, heading)
    print key_set, len(keys_odor_true)
    
    keys_odor = fad.get_keys_with_attr(dataset_control, ['odor_stimulus', 'visual_stimulus'], ['on', 'none'])
    keys_odor_true = []
    keys_odor_false = []
    for key in keys_odor:
        trajec = dataset_control.trajecs[key]
        if np.max(trajec.odor) > threshold_odor:
            keys_odor_true.append(key)
        else:
            keys_odor_false.append(key)
    key_set = 'none'
    key_sets.setdefault(key_set, keys_odor_true)
    heading = get_headings(dataset_control, config_control, keys_odor_true)
    data.setdefault(key_set, heading)
    print key_set, len(keys_odor_true)
    
            
    
        
    
    #colors = {'odor_true': 'red', 'odor_false': 'purple', 'noodor_true': 'black', 'noodor_false': 'blue'}
    colors = {'upwind': 'red', 'downwind': 'green', 'none': 'black'}
    
    make_histograms(config, data, colors, savename='visual_motion_heading_histogram.pdf')
    
    
    
    
def headings_parsed(dataset, config, visual_stimulus='none', direction='xy'):
    threshold_odor = 10
    key_sets = {}

    keys_odor = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['on', visual_stimulus])
    keys_odor_true = []
    keys_odor_false = []
    for key in keys_odor:
        trajec = dataset.trajecs[key]
        if np.max(trajec.odor) > threshold_odor:
            keys_odor_true.append(key)
        else:
            keys_odor_false.append(key)
    key_sets.setdefault('odor_true', keys_odor_true)
    key_sets.setdefault('odor_false', keys_odor_false)
            
    keys_noodor = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['none', visual_stimulus])
    keys_noodor_true = []
    keys_noodor_false = []
    for key in keys_noodor:
        trajec = dataset.trajecs[key]
        if np.max(trajec.odor) > threshold_odor:
            keys_noodor_true.append(key)
        else:
            keys_noodor_false.append(key)
    key_sets.setdefault('noodor_true', keys_noodor_true)
    key_sets.setdefault('noodor_false', keys_noodor_false)
    
    keys_afterodor = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['afterodor', visual_stimulus])
    keys_afterodor_true = []
    keys_afterodor_false = []
    for key in keys_afterodor:
        trajec = dataset.trajecs[key]
        #if np.max(trajec.odor) > threshold_odor:
        print trajec.timestamp_local_float - config.odor_stimulus['afterodor'][0]
        if trajec.timestamp_local_float - config.odor_stimulus['afterodor'][0] < 0.5:
            keys_afterodor_true.append(key)
        #else:
        #    keys_afterodor_false.append(key)
    key_sets.setdefault('afterodor_true', keys_afterodor_true)
    key_sets.setdefault('afterodor_false', keys_afterodor_false)
    
    data = {}
    colors = {}
    
    ranges = [[0, 0.5], [0.5, 5], [5,10000]]
    
    
    norm = matplotlib.colors.Normalize(0,1.5)
    cmap = matplotlib.cm.ScalarMappable(norm, 'jet')
    
    colorvals = ['red', 'green', 'purple']
        
    for i, r in enumerate(ranges):
        name = 'odor_true_' + str(r[0]) + '_' + str(r[1])
        
        if direction=='xy':
            headings = get_headings(dataset, config, keys_odor_true, odor_timerange=r, threshold_odor=threshold_odor)
        elif direction=='z_vel':
            headings = get_altitude_data(dataset, config, keys_odor_true, odor_timerange=r, threshold_odor=threshold_odor)
        elif direction=='z':
            headings = get_altitude_headings(dataset, config, keys_odor_true, odor_timerange=r, threshold_odor=threshold_odor)
        
        data.setdefault(name, headings)
        #colors.setdefault(name, cmap.to_rgba(r[0]))
        colors.setdefault(name, colorvals[i])
        
    if direction=='xy':
        headings = get_headings(dataset, config, keys_noodor_true, odor_timerange=None, threshold_odor=threshold_odor)
    elif direction=='z_vel':
        headings = get_altitude_data(dataset, config, keys_noodor_true, odor_timerange=None, threshold_odor=threshold_odor)
    elif direction=='z':
        headings = get_altitude_headings(dataset, config, keys_noodor_true, odor_timerange=None, threshold_odor=threshold_odor)
        
    data.setdefault('noodor', headings)
    colors.setdefault('noodor', 'black')
    
    if 0:
        if direction=='xy':
            headings = get_headings(dataset, config, keys_afterodor_true, odor_timerange=None, threshold_odor=threshold_odor)
        elif direction=='z':
            headings = get_altitude_data(dataset, config, keys_afterodor_true, odor_timerange=None, threshold_odor=threshold_odor)
        data.setdefault('afterodor', headings)
        colors.setdefault('afterodor', 'purple')
        
    savename='heading_histograms_parsed_' + direction + '_' + visual_stimulus + '.pdf'
    
    make_histograms(config, data, colors, direction=direction, savename=savename)
        
        
        
def super_simple(dataset, config, keys):
    threshold_odor = 10
    heading = get_headings(dataset, config, keys)
            
    data = {'only_data': heading}
    colors = {'only_data': 'black'}
        
    make_histograms(config, data, colors, savename='super_simple.pdf')
        
        
def make_histograms(config, data, colors, direction='xy', savename='heading_histogram.pdf'):

    print data
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)

    if direction=='xy':
        bins = np.linspace(-np.pi,np.pi,50)
    elif direction=='z_vel':
        bins = np.linspace(-.5,.5,50)
    elif direction=='z':
        bins = np.linspace(-2,2,50)
        
    
    
    fpl.histogram(ax, data.values(), bins=bins, bin_width_ratio=1, colors=colors.values(), edgecolor='none', bar_alpha=1, curve_fill_alpha=0.2, curve_line_alpha=1, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, smoothing_bins_to_exclude=[])
    
    if direction=='xy':
        xticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
    elif direction=='z_vel':
        xticks = [-.5,0,.5]
    elif direction=='z':
        xticks=[-2,2]
    
    ax.set_xlim(xticks[0], xticks[-1])
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    
    if direction=='xy':
        ax.set_xlabel('heading')
        xticklabels = ['-180', '-90', 'upwind', '90', '180']
        ax.set_xticklabels(xticklabels)
    elif direction=='z_vel':
        ax.set_xlabel('change in altitude')
    elif direction=='z':
        ax.set_xlabel('ratio of vertical to horizontal velocity')    
    
    ax.set_ylabel('occurences, normalized')

    path = config.path
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    fig_name_with_path = os.path.join(save_figure_path, savename)

    print 'SAVING TO: ', fig_name_with_path
    fig.savefig(fig_name_with_path, format='pdf')






def main(dataset, config):
    simple(dataset, config, visual_stimulus='none')
    






if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to data folder, where you have a configuration file")
    parser.add_option("--path_control", type="str", dest="path_control", default='',
                        help="path to data folder, where you have a configuration file")
    parser.add_option("--plot", type="str", dest="plot", default='',
                        help="which plot to do? options: visual_motion")
    parser.add_option("--direction", type="str", dest="direction", default='xy',
                        help="which direction - xy, or z")                  
    (options, args) = parser.parse_args()
    
    path = options.path    
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
    dataset = fad.load(culled_dataset_filename)
        
    path_control = options.path_control   
    if path_control != '': 
        analysis_configuration_control = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
        config_control = analysis_configuration.Config(path_control)
        culled_dataset_control_filename = os.path.join(path_control, config_control.culled_datasets_path, config_control.culled_dataset_name) 
        dataset_control = fad.load(culled_dataset_control_filename)
    
    if options.plot == 'visual_motion':
        visual_motion(dataset, config, dataset_control, config_control)
    elif options.plot == 'headings':
        headings_parsed(dataset, config, visual_stimulus='none', direction=options.direction)
    
    
    
    
    
    
    
    
