#!/usr/bin/env python
import sys, os
sys.path.append('../')
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

################# get trajectory keys #################

def get_keys(dataset):
    keys = dataset.trajecs.keys()
    return keys

################# plotting functions #######################

def plot_landing_histogram(config, dataset, save_figure_path=''):
    keys = get_keys(dataset)
    print 'number of keys: ', len(keys)
    if len(keys) < 1:
        print 'No data'
        return
        
    threshold_distance_min = 0.05
        
    keys_tmp = dataset.trajecs.keys()
    keys = []
    keys_not_near_post = []
    for key in keys_tmp:
        trajec = dataset.trajecs[key]
        if trajec.distance_to_post_min < threshold_distance_min:
            keys.append(key)
        else:
            keys_not_near_post.append(key)
            
    timestamps_not_landings = []
    timestamps_landings = []
    timestamps_not_near_post = []
    for i, key in enumerate(keys):
        trajec = dataset.trajecs[key]
        if 'landing' in trajec.post_behavior:
            timestamps_landings.append( trajec.timestamp_local_float )
        else:
            timestamps_not_landings.append( trajec.timestamp_local_float )
    for key in keys_not_near_post:
        trajec = dataset.trajecs[key]
        timestamps_not_near_post.append(trajec.timestamp_local_float)
    
    # shift so time continuous
    for i, t in enumerate(timestamps_not_landings):
        if t > 12: timestamps_not_landings[i] = t-24
    for i, t in enumerate(timestamps_landings):
        if t > 12: timestamps_landings[i] = t-24
    for i, t in enumerate(timestamps_not_near_post):
        if t > 12: timestamps_not_near_post[i] = t-24
    
            
    timestamps_not_landings = np.array(timestamps_not_landings)
    timestamps_landings = np.array(timestamps_landings)
    timestamps_not_near_post = np.array(timestamps_not_near_post)
    
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    
    nbins = 36 # note: if show_smoothed=True with default butter filter, nbins needs to be > ~15 
    bins1 = np.linspace(-12,-0.21,int(nbins/2.),endpoint=True)
    bins2 = np.linspace(-.21,12,int(nbins/2.),endpoint=True)
    #bins = [16.05-24, 17.05-24, 18.05-24, 19.05-24, 20.05-24, 21.05-24, 22.05-24, 23.05-24, 0.05, 1.05, 2.05, 3.05, 4.05, 5.05, 6.05, 7.05, 8.05, 9.05, 10.05]
    bins = np.arange(16.05-24,10.05, 0.5)
    
    
    data = [timestamps_landings, timestamps_not_landings, timestamps_not_near_post]
    if save_figure_path == '':
        save_figure_path = os.path.join(config.path, config.figure_path, 'activity/')
    colors = ['black', 'green']
    
    fpl.histogram_stack(ax, data, bins=bins, bin_width_ratio=0.8, colors=['green', 'black', 'gray'], edgecolor='none', normed=True)
    #fpl.histogram(ax, data, bins=bins, bin_width_ratio=0.8, colors=colors, edgecolor='none', bar_alpha=1, curve_fill_alpha=0.4, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=False, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False)
    
    if 'on' in config.odor_stimulus.keys():
        odor_on = config.odor_stimulus['on']
        if type(odor_on[0]) is not list: 
            ax.fill_between(np.array(odor_on), -10, 10, color='red', alpha=0.25, zorder=-10, edgecolor='none')
        else:
            for vals in odor_on:
                odor_start = vals[0]
                odor_end = vals[-1]
                if odor_start > 12:
                    odor_start -= 24
                if odor_end > 12:
                    odor_end -= 24
                print 'odor on for time ranges: '
                print odor_start, odor_end
                ax.fill_between(np.array([odor_start, odor_end]), -10, 10, color='red', alpha=0.25, zorder=-10, edgecolor='none')
    
    #xticks = [0,6,12,18,24]
    xticks = [-12,-6,0,6,12]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    
    ax.set_xlabel('Time of day, hours')
    ax.set_ylabel('Occurences, normalized')
    ax.set_title('Landings: green -- Nonlandings: black')
    
    fig.set_size_inches(4,4)
    if save_figure_path == '':
        save_figure_path = os.path.join(config.path, config.figure_path, 'activity/')
    figname = save_figure_path + 'landing_histogram' + '.pdf'
    fig.savefig(figname, format='pdf')

    
    
def main(config, culled_dataset, save_figure_path=''):
    print
    print 'Plotting activity histogram'
    plot_landing_histogram(config, culled_dataset, save_figure_path=save_figure_path)

if __name__ == '__main__':
    config = analysis_configuration.Config()
    culled_dataset = fad.load('../' + config.culled_datasets_path + config.culled_dataset_name)
    
    main(culled_dataset, save_figure_path='../../figures/activity_histograms/')
    


