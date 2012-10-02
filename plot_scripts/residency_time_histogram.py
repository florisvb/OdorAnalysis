#!/usr/bin/env python
import sys, os
sys.path.append('../')
sys.path.append('../analysis_modules')
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

import odor_packet_analysis as opa

################# plotting functions #######################

def plot_residency_time_histogram(config, dataset, save_figure_path=''):
    odor_stimulus = 'on'
    threshold_distance_min = 0.05
    
    
    # no odor
    keys_odor_off = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=10, threshold_distance=-1, odor_stimulus='none', upwind_only=True, threshold_distance_min=0.1, odor=True, post_behavior='landing')
    residency_time_odor_off = []
    for key in keys_odor_off:
        trajec = dataset.trajecs[key]
        if trajec.residency_time is not None:
            residency_time_odor_off.append(trajec.residency_time)
            
    # odor on, odor experienced
    keys_odor_on_true = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=10, threshold_distance=-1, odor_stimulus='on', upwind_only=True, threshold_distance_min=0.1, odor=True, post_behavior='landing')
    residency_time_odor_on_true = []
    for key in keys_odor_on_true:
        trajec = dataset.trajecs[key]
        if trajec.residency_time is not None:
            residency_time_odor_on_true.append(trajec.residency_time)
        
    # odor on, odor not experienced
    keys_odor_on_false = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=10, threshold_distance=-1, odor_stimulus='on', upwind_only=False, threshold_distance_min=0.1, odor=False, post_behavior='landing')
    residency_time_odor_on_false = []
    for key in keys_odor_on_false:
        trajec = dataset.trajecs[key]
        if trajec.residency_time is not None:
            residency_time_odor_on_false.append(trajec.residency_time)
    

    data = [np.array(residency_time_odor_off), np.array(residency_time_odor_on_true), np.array(residency_time_odor_on_false)]
    colors = ['black', 'red', 'blue']
    
    nbins = 30 # note: if show_smoothed=True with default butter filter, nbins needs to be > ~15 

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.histogram(ax, data, bins=nbins, bin_width_ratio=0.8, colors=colors, edgecolor='none', bar_alpha=1, curve_fill_alpha=0.4, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False)
    
    #xticks = [0,.02,.04,.06,.08,.15]
    fpl.adjust_spines(ax, ['left', 'bottom'])
    #ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xlabel('residency time, frames')
    ax.set_ylabel('Occurences, normalized')
    ax.set_title('Residency time')
    
    
    if save_figure_path == '':
        save_figure_path = os.path.join(config.path, config.figure_path, 'activity/')
    fig.set_size_inches(8,8)
    figname = save_figure_path + 'residency_time_on_post_histogram' + '.pdf'
    fig.savefig(figname, format='pdf')

    
    
def main(config, dataset, save_figure_path=''):
    print
    print 'Plotting distance to post histogram'
    plot_residency_time_histogram(config, dataset, save_figure_path=save_figure_path)

if __name__ == '__main__':
    config = analysis_configuration.Config()
    culled_dataset = fad.load('../' + config.culled_datasets_path + config.culled_dataset_name)
    
    main(culled_dataset, save_figure_path='../../figures/activity_histograms/')
