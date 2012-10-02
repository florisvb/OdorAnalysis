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

def plot_distance_histogram(config, dataset, save_figure_path=''):
    
    # no odor
    keys_no_odor = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=10, odor_stimulus='none', threshold_distance_min=0.1, odor=True)
    data_no_odor = []
    for key in keys_no_odor:
        trajec = dataset.trajecs[key]
        if np.max(trajec.distance_to_post) > 0.1:
            for f, d in enumerate(trajec.distance_to_post):
                if trajec.positions[f,0] > 0:
                    data_no_odor.append(d)
        
    # odor on, odor experienced
    keys_odor_on_true = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=10, odor_stimulus='on', threshold_distance_min=0.1, odor=True)
    data_odor_on_true = []
    for key in keys_odor_on_true:
        trajec = dataset.trajecs[key]
        if np.max(trajec.distance_to_post) > 0.1:
            for f, d in enumerate(trajec.distance_to_post):
                if trajec.positions[f,0] > 0:
                    data_odor_on_true.append(d)
    
    print len(data_no_odor)
    print len(data_odor_on_true)
    
    data = [np.array(data_no_odor), np.array(data_odor_on_true)]
    colors = ['black', 'red']
    nbins = 30
    bins = np.linspace(0,0.1,nbins)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.histogram(ax, data, bins=bins, bin_width_ratio=0.8, colors=colors, edgecolor='none', bar_alpha=1, curve_fill_alpha=0.4, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=True, exponential_histogram=False, n_bootstrap_samples=10000, smoothing_range=[0.005,0.1])
    
    xticks = [0,.02,.04,.06,.08,.1]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xlabel('Distance to post, m')
    ax.set_ylabel('Occurences, normalized')
    title_text = 'Distance to post. red (upwind, odor) N = ' + str(len(keys_odor_on_true)) + '; black (upwind, no odor) N = ' + str(len(keys_no_odor))
    ax.set_title(title_text)
    
    
    if save_figure_path == '':
        save_figure_path = os.path.join(config.path, config.figure_path, 'activity/')
    fig.set_size_inches(8,8)
    figname = save_figure_path + 'distance_to_post_histogram' + '.pdf'
    fig.savefig(figname, format='pdf')

def plot_distance_histogram_2(config, dataset, save_figure_path=''):
    
    # no odor
    keys_no_odor = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=10, odor_stimulus='none', threshold_distance_min=0.1, odor=False)
    data_no_odor = []
    for key in keys_no_odor:
        trajec = dataset.trajecs[key]
        if np.max(trajec.distance_to_post) > 0.1:
            data_no_odor.extend(trajec.distance_to_post)
        
    # odor on, no odor experienced
    keys_odor_on_false = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=10, odor_stimulus='on', threshold_distance_min=0.1, odor=False)
    data_odor_on_false = []
    for key in keys_odor_on_false:
        trajec = dataset.trajecs[key]
        if np.max(trajec.distance_to_post) > 0.1:
            data_odor_on_false.extend(trajec.distance_to_post)
        
    data = [np.array(data_no_odor), np.array(data_odor_on_false)]
    colors = ['black', 'blue']
    nbins = 30
    bins = np.linspace(0,0.1,nbins)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.histogram(ax, data, bins=bins, bin_width_ratio=0.8, colors=colors, edgecolor='none', bar_alpha=1, curve_fill_alpha=0.4, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=True, exponential_histogram=False, n_bootstrap_samples=10000, smoothing_range=[0.005,0.1])
    
    xticks = [0,.02,.04,.06,.08,.1]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xlabel('Distance to post, m')
    ax.set_ylabel('Occurences, normalized')
    title_text = 'Distance to post. blue (odor, no odor) N = ' + str(len(keys_odor_on_false)) + '; black (no odor, no odor) N = ' + str(len(keys_no_odor))
    ax.set_title(title_text)
    
    
    if save_figure_path == '':
        save_figure_path = os.path.join(config.path, config.figure_path, 'activity/')
    fig.set_size_inches(8,8)
    figname = save_figure_path + 'distance_to_post_histogram_2' + '.pdf'
    fig.savefig(figname, format='pdf')

    
def main(config, dataset, save_figure_path=''):
    print
    print 'Plotting distance to post histogram'
    plot_distance_histogram(config, dataset, save_figure_path=save_figure_path)
    plot_distance_histogram_2(config, dataset, save_figure_path=save_figure_path)

if __name__ == '__main__':
    config = analysis_configuration.Config()
    culled_dataset = fad.load('../' + config.culled_datasets_path + config.culled_dataset_name)
    
    main(culled_dataset, save_figure_path='../../figures/activity_histograms/')
