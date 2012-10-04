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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

def in_range(val, minmax):
    if val > minmax[0] and val < minmax[1]:
        return True
    else:
        return False

def get_headings(dataset, config, keys, odor_timerange=None, threshold_odor=10):
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
            if np.abs(trajec.positions[f,2]) > 0.1: # skip low and high points
                use_frame = False
            if use_frame:
                frames_to_use.append(f)
        headings.extend(trajec.heading_smooth[frames_to_use].tolist())

    headings = np.array(headings)
    
    headings_less_than_zero = np.where(headings < 0)[0]
    headings_greater_than_zero = np.where(headings > 0)[0]
    
    headings[headings_less_than_zero] = headings[headings_less_than_zero] + np.pi
    headings[headings_greater_than_zero] = headings[headings_greater_than_zero] - np.pi
    
    return headings    
    
    
    

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
    heading = get_headings(dataset, config, keys_odor_true)
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
    heading = get_headings(dataset, config, keys_odor_true)
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
    
    
    
    
def headings_parsed(dataset, config, visual_stimulus='none'):
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
    
    data = {}
    colors = {}
    
    ranges = [[0, 0.5], [0.5, 5]]
    
    
    norm = matplotlib.colors.Normalize(0,1.5)
    cmap = matplotlib.cm.ScalarMappable(norm, 'jet')
    
    colorvals = ['red', 'green']
        
    for i, r in enumerate(ranges):
        name = 'odor_true_' + str(r[0]) + '_' + str(r[1])
        headings = get_headings(dataset, config, keys_odor_true, odor_timerange=r, threshold_odor=threshold_odor)
        data.setdefault(name, headings)
        #colors.setdefault(name, cmap.to_rgba(r[0]))
        colors.setdefault(name, colorvals[i])
        
    headings = get_headings(dataset, config, keys_noodor_true, odor_timerange=None, threshold_odor=threshold_odor)
    data.setdefault('noodor', headings)
    colors.setdefault('noodor', 'black')
    
    make_histograms(config, data, colors)
        
        
        
def super_simple(dataset, config, keys):
    threshold_odor = 10
    heading = get_headings(dataset, config, keys)
            
    data = {'only_data': heading}
    colors = {'only_data': 'black'}
        
    make_histograms(config, data, colors, savename='super_simple.pdf')
        
        
def make_histograms(config, data, colors, savename='heading_histogram.pdf'):

    print data
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)

    bins = 100
    
    fpl.histogram(ax, data.values(), bins=bins, bin_width_ratio=1, colors=colors.values(), edgecolor='none', bar_alpha=1, curve_fill_alpha=0.2, curve_line_alpha=1, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False)

    xticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    xticklabels = ['-180', '-90', 'upwind', '90', '180']
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('heading')
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
        headings_parsed(dataset, config, visual_stimulus='none')
    
    
    
    
    
    
    
    
