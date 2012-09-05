#!/usr/bin/env python
import sys, os
sys.path.append('../')
import flydra_analysis_tools as fat
import fly_plot_lib
fly_plot_lib.set_params.pdf()
from matplotlib.backends.backend_pdf import PdfPages
import fly_plot_lib.plot as fpl
fad = fat.flydra_analysis_dataset
dac = fat.dataset_analysis_core
fap = fat.flydra_analysis_plot
tac = fat.trajectory_analysis_core

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches

import odor_packet_analysis as opa

import xyz_axis_grid

################# get trajectory keys #################

def get_keys(dataset):
    keys = dataset.trajecs.keys()
    return keys
    

################# plot all heatmaps together ###############

def plot_all_heatmaps(config, dataset, save_figure_path='', figname=None, keys=None, frames=None, title=None):
    
    ax_xy, ax_xz, ax_yz = xyz_axis_grid.get_axes()

    plot_heatmap(config, dataset, axis='xy', save_figure_path='', figname=None, keys=keys, frames=frames, ax=ax_xy, save=False)
    plot_heatmap(config, dataset, axis='xz', save_figure_path='', figname=None, keys=keys, frames=frames, ax=ax_xz, save=False)
    plot_heatmap(config, dataset, axis='yz', save_figure_path='', figname=None, keys=keys, frames=frames, ax=ax_yz, save=False)
    
    xyz_axis_grid.set_spines_and_labels(ax_xy, ax_xz, ax_yz)
    
    if title is not None:
        plt.suptitle(title)

    if figname is None:
        figname = save_figure_path + 'heatmap_' + 'all_axes' + '.pdf'
    else:
        figname = save_figure_path + figname
    plt.savefig(figname, format='pdf')

################# plotting functions #######################

def plot_heatmap(config, dataset, axis='xy', save_figure_path='', figname=None, keys=None, frames=None, ax=None, save=True):
    if keys is None:
        keys = get_keys(dataset)
    print 'plotting heatmap, axis: ', axis
    print 'number of keys: ', len(keys)
    if len(keys) < 1:
        print 'No data'
        return
        
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    colornorm = [0,len(keys)/2.]
    fap.heatmap(ax, dataset, axis=axis, keys=keys, xticks=config.ticks['x'], yticks=config.ticks['y'], zticks=config.ticks['z'], rticks=config.ticks['r'], colornorm=colornorm, normalize_for_speed=False, frame_list=frames)
    
    
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
        post = patches.Rectangle([-1*config.post_radius+config.post_center[1], config.ticks['z'][0]], config.post_radius*2, height, color='black')
        artists.append(post)
    if axis == 'rz':
        ax.set_xlabel('r axis')
        ax.set_ylabel('z axis')
        post = patches.Rectangle([0.0,config.ticks['z'][0]], config.post_radius, height, color='black')
        artists.append(post)
    
    if artists is not None:
        for artist in artists:
            ax.add_artist(artist)
    
    if save:
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
    if len(dataset_in_odor.trajecs.keys()) > 0:
        if 1:
            plot_all_heatmaps(config, dataset_in_odor, save_figure_path=save_figure_path, figname='heatmap_odor_all_axes.pdf', keys=None, frames=None)
        if 0:
            plot_heatmap(config, dataset_in_odor, 'xy', save_figure_path=save_figure_path, figname='heatmap_odor_xy.pdf')
            plot_heatmap(config, dataset_in_odor, 'yz', save_figure_path=save_figure_path, figname='heatmap_odor_yz.pdf')
            plot_heatmap(config, dataset_in_odor, 'xz', save_figure_path=save_figure_path, figname='heatmap_odor_xz.pdf')
            plot_heatmap(config, dataset_in_odor, 'rz', save_figure_path=save_figure_path, figname='heatmap_odor_rz.pdf')

    # not in odor
    print
    print 'No odor: '
    dataset_no_odor = fad.make_dataset_with_attribute_filter(culled_dataset, 'odor_stimulus', 'none')
    if len(dataset_no_odor.trajecs.keys()) > 0:
        if 1:
            plot_all_heatmaps(config, dataset_no_odor, save_figure_path=save_figure_path, figname='heatmap_no_odor_all_axes.pdf', keys=None, frames=None)
        if 0:
            plot_heatmap(config, dataset_no_odor, 'xy', save_figure_path=save_figure_path, figname='heatmap_no_odor_xy.pdf')
            plot_heatmap(config, dataset_no_odor, 'yz', save_figure_path=save_figure_path, figname='heatmap_no_odor_yz.pdf')
            plot_heatmap(config, dataset_no_odor, 'xz', save_figure_path=save_figure_path, figname='heatmap_no_odor_xz.pdf')
            plot_heatmap(config, dataset_no_odor, 'rz', save_figure_path=save_figure_path, figname='heatmap_no_odor_rz.pdf')
    
    # pulse odor
    print
    print 'Pulsing odor: '
    dataset_pulsing_odor = fad.make_dataset_with_attribute_filter(culled_dataset, 'odor_stimulus', 'pulsing')
    if len(dataset_pulsing_odor.trajecs.keys()) > 0:
        if 1:
            plot_all_heatmaps(config, dataset_pulsing_odor, save_figure_path=save_figure_path, figname='heatmap_pulsing_odor_all_axes.pdf', keys=None, frames=None)
        if 0:
            plot_heatmap(config, dataset_pulsing_odor, 'xy', save_figure_path=save_figure_path, figname='heatmap_pulsing_odor_xy.pdf')
            plot_heatmap(config, dataset_pulsing_odor, 'yz', save_figure_path=save_figure_path, figname='heatmap_pulsing_odor_yz.pdf')
            plot_heatmap(config, dataset_pulsing_odor, 'xz', save_figure_path=save_figure_path, figname='heatmap_pulsing_odor_xz.pdf')
            plot_heatmap(config, dataset_pulsing_odor, 'rz', save_figure_path=save_figure_path, figname='heatmap_pulsing_odor_rz.pdf')
    
    # after odor
    print
    print 'After odor: '
    dataset_after_odor = fad.make_dataset_with_attribute_filter(culled_dataset, 'odor_stimulus', 'afterodor')
    if len(dataset_after_odor.trajecs.keys()) > 0:
        if 1:
            plot_all_heatmaps(config, dataset_after_odor, save_figure_path=save_figure_path, figname='heatmap_after_odor_all_axes.pdf', keys=None, frames=None)
        if 0:
            plot_heatmap(config, dataset_after_odor, 'xy', save_figure_path=save_figure_path, figname='heatmap_after_odor_xy.pdf')
            plot_heatmap(config, dataset_after_odor, 'yz', save_figure_path=save_figure_path, figname='heatmap_after_odor_yz.pdf')
            plot_heatmap(config, dataset_after_odor, 'xz', save_figure_path=save_figure_path, figname='heatmap_after_odor_xz.pdf')
            plot_heatmap(config, dataset_after_odor, 'rz', save_figure_path=save_figure_path, figname='heatmap_after_odor_rz.pdf')
        
        
        
    # only flies that passed through odor (prior to post)
    print 
    print 'Flies that passed through odor: '
    keys = opa.get_keys_with_odor_before_post(config, culled_dataset, threshold_odor=50, threshold_distance=0.01, odor_stimulus='pulsing', upwind_only=True)
    frames = opa.get_frames_after_odor(culled_dataset, keys, frames_to_show_before_odor=25)
    if len(keys) > 0:
        if 1:
            plot_all_heatmaps(config, culled_dataset, save_figure_path=save_figure_path, figname='heatmap_flies_in_plume_all_axes.pdf', keys=keys, frames=frames)
        if 0:
            plot_heatmap(config, culled_dataset, 'xy', keys=keys, frames=frames, save_figure_path=save_figure_path, figname='heatmap_flies_in_odor_plume_xy.pdf')
            plot_heatmap(config, culled_dataset, 'yz', keys=keys, frames=frames, save_figure_path=save_figure_path, figname='heatmap_flies_in_odor_plume_yz.pdf')
            plot_heatmap(config, culled_dataset, 'xz', keys=keys, frames=frames, save_figure_path=save_figure_path, figname='heatmap_flies_in_odor_plume_xz.pdf')
            plot_heatmap(config, culled_dataset, 'rz', keys=keys, frames=frames, save_figure_path=save_figure_path, figname='heatmap_flies_in_odor_plume_rz.pdf')
    
def pdf_book(config, dataset, save_figure_path=''):
    figure_path = save_figure_path
    pdf_name_with_path = os.path.join(save_figure_path, 'heatmap_book.pdf')
    pp = PdfPages(pdf_name_with_path)
    
    key_set = {}
    for odor_stimulus in config.odor_stimulus.keys():
        key_set.setdefault(odor_stimulus, fad.get_keys_with_attr(dataset, 'odor_stimulus', odor_stimulus))

    for odor_stimulus, keys in key_set.items():
        print 'Odor Book, Chapter: ', odor_stimulus
        fig = plt.figure()
        figname = 'heatmap_' + odor_stimulus + '_odor_all_axes.pdf'
        title = 'odor ' + odor_stimulus
        plot_all_heatmaps(config, dataset, save_figure_path=save_figure_path, figname=figname, title=title.title(), keys=keys)
        pp.savefig()
        plt.close('all')
        
    pp.close()
    
    
    
if __name__ == '__main__':
    config = analysis_configuration.Config()
    culled_dataset = fad.load('../' + config.culled_datasets_path + config.culled_dataset_name)
    
    pdf_book(config, culled_dataset, save_figure_path='../../figures/heatmaps/')
    


