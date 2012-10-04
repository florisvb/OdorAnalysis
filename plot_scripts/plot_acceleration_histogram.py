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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

def in_range(val, minmax):
    if val > minmax[0] and val < minmax[1]:
        return True
    else:
        return False

    

def get_accelerations(dataset, config, keys, odor_timerange=None, threshold_odor=10, behavior='surge'):
    accels_xy = []
    accels_z = []
    for key in keys:
        trajec = dataset.trajecs[key]
        accelerations_xy = trajec.speed_xy#floris_math.diffa(trajec.speed_xy)*trajec.fps
        accelerations_z = trajec.velocities[:,2]#floris_math.diffa(trajec.velocities[:,2])*trajec.fps
        
        frames_where_odor = np.where(trajec.odor > threshold_odor)[0]
        saccade_frames = [item for sublist in trajec.saccades for item in sublist]
        frames_to_use = []
        for f, s in enumerate(trajec.speed):
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
                
            #if np.abs(trajec.heading_smooth[f]) > np.pi/4 and np.abs(trajec.heading_smooth[f]) < np.pi-np.pi/4.:
            #    use_frame = False
            
            if behavior=='surge':
                if np.abs(trajec.heading_smooth[f]) < np.pi/4.:
                    use_frame = False
            if behavior=='cast':
                if np.abs(trajec.heading_smooth[f]) > np.pi/4. or np.abs(trajec.heading_smooth[f]) > np.pi-np.pi/4.:
                    use_frame = False
                
            if use_frame:
                frames_to_use.append(f)
                
        accels_xy.extend(accelerations_xy[frames_to_use].tolist())
        accels_z.extend(accelerations_z[frames_to_use].tolist())

    accels_xy = np.array(accels_xy)
    accels_z = np.array(accels_z)
    
    return accels_xy, accels_z
    
    
    
def accelerations_parsed(dataset, config, visual_stimulus='none', axis='xy'):
    threshold_odor = 10
    key_sets = {}

    keys_odor = fad.get_keys_with_attr(dataset, ['odor_stimulus'], ['on'])
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
            
    keys_noodor = fad.get_keys_with_attr(dataset, ['odor_stimulus'], ['none'])
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
    
    ranges = [[0, 0.1], [0.1, 10000]]
    
    norm = matplotlib.colors.Normalize(0,3)
    cmap = matplotlib.cm.ScalarMappable(norm, 'jet')
    
    colorvals = ['red', 'blue']
        
    for i, r in enumerate(ranges):
        name = 'odor_true_' + str(r[0]) + '_' + str(r[1])
        accels_xy, accels_z = get_accelerations(dataset, config, keys_odor_true, odor_timerange=r, threshold_odor=threshold_odor)
        
        if axis == 'xy':
            data.setdefault(name, accels_xy)
        elif axis == 'z':
            data.setdefault(name, accels_z)
        
        #colors.setdefault(name, cmap.to_rgba(r[0]))
        colors.setdefault(name, colorvals[i])
        
    accels_xy, accels_z = get_accelerations(dataset, config, keys_noodor_true, odor_timerange=None, threshold_odor=threshold_odor)
    if axis == 'xy':
        data.setdefault('noodor', accels_xy)
    elif axis == 'z':
        data.setdefault('noodor', accels_z)
            
    colors.setdefault('noodor', 'black')
    
    figname = axis + '_speed_histogram.pdf'
    make_histograms(config, data, colors, axis, savename=figname)
    
    
    
    
        
def super_simple(dataset, config, keys=None):
    threshold_odor = 10
    keys = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'none')
    altitudes = get_altitudes(dataset, config, keys)
            
    data = {'only_data': altitudes}
    colors = {'only_data': 'black'}
        
    make_histograms(config, data, colors, savename='altitude_simple_histogram.pdf')
        
        
def make_histograms(config, data, colors, axis, savename='histogram.pdf'):

    for key, item in data.items():
        print key, len(item)
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    
    if axis=='z':
        bins = np.linspace(-0.5,0.5,50,endpoint=True)
        bins_to_exclude = [24]
    elif axis=='xy':
        bins = np.linspace(0.05,1,50,endpoint=True)
        bins_to_exclude = []
    
    fpl.histogram(ax, data.values(), bins=bins, bin_width_ratio=1, colors=colors.values(), edgecolor='none', bar_alpha=1, curve_fill_alpha=0.2, curve_line_alpha=1, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, smoothing_bins_to_exclude=bins_to_exclude)

    if axis=='z':
        xticks = [-0.5, -.25, 0, .25, .5]
    elif axis == 'xy':
        xticks = [0,.2,.4,.6,.8,1.]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    #ax.set_xticklabels(xticklabels)
    ax.set_xlabel('speed in ' + axis)
    ax.set_ylabel('occurences, normalized')
    
    ax.set_xlim(xticks[0], xticks[-1])
    
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
    
    
    if options.plot == 'simple':
        super_simple(dataset, config)
    
    
    if options.plot == 'parsed_xy':
        accelerations_parsed(dataset, config)
        
    if options.plot == 'parsed_z':
        accelerations_parsed(dataset, config, axis='z')
    
    
    
