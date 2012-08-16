#!/usr/bin/env python
import sys
sys.path.append('../')
import flydra_analysis_tools as fat
from flydra_analysis_tools import floris_plot_lib as fpl
fad = fat.flydra_analysis_dataset
dac = fat.dataset_analysis_core
fap = fat.flydra_analysis_plot
tac = fat.trajectory_analysis_core

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

################# get trajectory keys #################

def get_keys(dataset):
    keys = dataset.trajecs.keys()
    return keys

################# plotting functions #######################

def plot_heatmap(config, dataset, axis='xy', save_figure_path='', figname=None):
    keys = get_keys(dataset)
    print 'plotting heatmap, axis: ', axis
    print 'number of keys: ', len(keys)
    if len(keys) < 1:
        print 'No data'
        return
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    colornorm = [0,len(keys)/2.]
    fap.heatmap(ax, dataset, axis=axis, keys=keys, xticks=config.ticks['x'], yticks=config.ticks['y'], zticks=config.ticks['z'], rticks=config.ticks['r'], colornorm=colornorm, normalize_for_speed=True)
    
    
    height = config.post_center[2]-config.ticks['z'][0]
    
    artists = []
    if axis == 'xy':
        ax.set_xlabel('x axis, upwind negative')
        ax.set_ylabel('y axis')
        post = patches.Circle(config.post_center[0:2], config.post_radius, color='black')
        artists.append(post)
    if axis == 'xz':
        ax.set_xlabel('x axis, upwind negative')
        ax.set_ylabel('z axis')
        post = patches.Rectangle([-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black')
        artists.append(post)
    if axis == 'yz':
        ax.set_xlabel('y axis')
        ax.set_ylabel('z axis')
        post = patches.Rectangle([-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black')
        artists.append(post)
    if axis == 'rz':
        ax.set_xlabel('r axis')
        ax.set_ylabel('z axis')
        post = patches.Rectangle([0.0,config.ticks['z'][0]], config.post_radius, height, color='black')
        artists.append(post)
    
    if artists is not None:
        for artist in artists:
            ax.add_artist(artist)
    
    if figname is None:
        figname = save_figure_path + 'heatmap_' + axis + '.pdf'
    else:
        figname = save_figure_path + figname
    fig.savefig(figname, format='pdf')
    
def main(config, culled_dataset, save_figure_path=''):
    print
    print 'Plotting heatmaps'
    
    # in odor
    print
    print 'Odor: '
    dataset_in_odor = fad.make_dataset_with_attribute_filter(culled_dataset, 'odor_stimulus', 'on')
    plot_heatmap(config, dataset_in_odor, 'xy', save_figure_path=save_figure_path, figname='heatmap_odor_xy.pdf')
    plot_heatmap(config, dataset_in_odor, 'yz', save_figure_path=save_figure_path, figname='heatmap_odor_yz.pdf')
    plot_heatmap(config, dataset_in_odor, 'xz', save_figure_path=save_figure_path, figname='heatmap_odor_xz.pdf')
    plot_heatmap(config, dataset_in_odor, 'rz', save_figure_path=save_figure_path, figname='heatmap_odor_rz.pdf')

    # not in odor
    print
    print 'No odor: '
    dataset_no_odor = fad.make_dataset_with_attribute_filter(culled_dataset, 'odor_stimulus', 'none')
    plot_heatmap(config, dataset_no_odor, 'xy', save_figure_path=save_figure_path, figname='heatmap_no_odor_xy.pdf')
    plot_heatmap(config, dataset_no_odor, 'yz', save_figure_path=save_figure_path, figname='heatmap_no_odor_yz.pdf')
    plot_heatmap(config, dataset_no_odor, 'xz', save_figure_path=save_figure_path, figname='heatmap_no_odor_xz.pdf')
    plot_heatmap(config, dataset_no_odor, 'rz', save_figure_path=save_figure_path, figname='heatmap_no_odor_rz.pdf')
    
    # pulse odor
    print
    print 'Pulsing odor: '
    dataset_pulsing_odor = fad.make_dataset_with_attribute_filter(culled_dataset, 'odor_stimulus', 'pulsing')
    plot_heatmap(config, dataset_pulsing_odor, 'xy', save_figure_path=save_figure_path, figname='heatmap_pulsing_odor_xy.pdf')
    plot_heatmap(config, dataset_pulsing_odor, 'yz', save_figure_path=save_figure_path, figname='heatmap_pulsing_odor_yz.pdf')
    plot_heatmap(config, dataset_pulsing_odor, 'xz', save_figure_path=save_figure_path, figname='heatmap_pulsing_odor_xz.pdf')
    plot_heatmap(config, dataset_pulsing_odor, 'rz', save_figure_path=save_figure_path, figname='heatmap_pulsing_odor_rz.pdf')
    
    # after odor
    print
    print 'After odor: '
    dataset_after_odor = fad.make_dataset_with_attribute_filter(culled_dataset, 'odor_stimulus', 'afterodor')
    plot_heatmap(config, dataset_after_odor, 'xy', save_figure_path=save_figure_path, figname='heatmap_after_odor_xy.pdf')
    plot_heatmap(config, dataset_after_odor, 'yz', save_figure_path=save_figure_path, figname='heatmap_after_odor_yz.pdf')
    plot_heatmap(config, dataset_after_odor, 'xz', save_figure_path=save_figure_path, figname='heatmap_after_odor_xz.pdf')
    plot_heatmap(config, dataset_after_odor, 'rz', save_figure_path=save_figure_path, figname='heatmap_after_odor_rz.pdf')
    
if __name__ == '__main__':
    config = analysis_configuration.Config()
    culled_dataset = fad.load('../' + config.culled_datasets_path + config.culled_dataset_name)
    
    main(config, culled_dataset, save_figure_path='../../figures/heatmaps/')
    


