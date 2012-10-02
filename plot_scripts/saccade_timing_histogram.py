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

import help_functions as hf

def get_time_to_saccade(dataset, keys, threshold_odor=25):
    
    time_to_saccade = []
    for key in keys:
        trajec = dataset.trajecs[key]
        frames_in_odor = np.where(trajec.odor > threshold_odor)[0].tolist()
        if len(frames_in_odor) <= 0:
            continue
        odor_blocks = hf.find_continuous_blocks(frames_in_odor, 5)
        for block in odor_blocks:
            if len(block) < 1:
                continue
            for sac in trajec.saccades:
                if sac[0] > block[-1]:
                    t_to_sac = (sac[0] - block[-1])/trajec.fps
                    time_to_saccade.append(t_to_sac)
                    break
    return time_to_saccade

################# plotting functions #######################

def plot_time_to_saccade_histogram(config, dataset, save_figure_path=''):
    threshold_odor = 100
    
    # no odor
    keys_odor_off = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=threshold_odor, threshold_distance=-1, odor_stimulus='none', upwind_only=True, threshold_distance_min=0.1, odor=True)
    time_to_saccade_odor_off = get_time_to_saccade(dataset, keys_odor_off, threshold_odor=threshold_odor)
            
    # odor on, odor experienced
    keys_odor_on_true = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=threshold_odor, threshold_distance=-1, odor_stimulus='on', upwind_only=True, threshold_distance_min=0.1, odor=True)
    time_to_saccade_odor_on_true = get_time_to_saccade(dataset, keys_odor_on_true, threshold_odor=threshold_odor)
    
    # odor on, odor not experienced
    keys_odor_on_false = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=threshold_odor, threshold_distance=-1, odor_stimulus='on', upwind_only=False, threshold_distance_min=0.1, odor=False)
    keys_ok = []
    for key in keys_odor_on_false:
        trajec = dataset.trajecs[key]
        if np.max(trajec.odor) > 0.01:
            keys_ok.append(key)
    print 'control odor on: ', len(keys_ok)
    time_to_saccade_odor_on_false = get_time_to_saccade(dataset, keys_ok, threshold_odor=0.001)
        

    data = [np.array(time_to_saccade_odor_off), np.array(time_to_saccade_odor_on_true)]
    print [len(d) for d in data]
    
    colors = ['black', 'red']
    
    nbins = 25 # note: if show_smoothed=True with default butter filter, nbins needs to be > ~15 
    bins = np.linspace(0,1,nbins)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.histogram(ax, data, bins=bins, bin_width_ratio=0.8, colors=colors, edgecolor='none', bar_alpha=1, curve_fill_alpha=0.4, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=True, exponential_histogram=False)
    
    xticks = [0,0.2,0.4,0.5, 1.]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xlabel('Time to saccade after leaving odor, secs')
    ax.set_ylabel('Occurences, normalized')
    ax.set_title('Time to saccade')
    
    
    if save_figure_path == '':
        save_figure_path = os.path.join(config.path, config.figure_path, 'activity/')
    fig.set_size_inches(8,8)
    figname = save_figure_path + 'time_to_saccade_histogram' + '.pdf'
    fig.savefig(figname, format='pdf')

    
    
def main(config, dataset, save_figure_path=''):
    print
    print 'Plotting time to saccade histogram'
    plot_time_to_saccade_histogram(config, dataset, save_figure_path=save_figure_path)

if __name__ == '__main__':
    config = analysis_configuration.Config()
    culled_dataset = fad.load('../' + config.culled_datasets_path + config.culled_dataset_name)
    
    main(culled_dataset, save_figure_path='../../figures/activity_histograms/')
