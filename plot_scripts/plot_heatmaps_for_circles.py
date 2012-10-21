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

def plot_all_heatmaps(config, dataset, save_figure_path='', figname=None, keys=None, frames=None, title=None, save=False):
    
    ax_xy, ax_xz, ax_yz = xyz_axis_grid.get_axes()

    plot_heatmap(config, dataset, axis='xy', save_figure_path='', figname=None, keys=keys, frames=frames, ax=ax_xy, save=False)
    plot_heatmap(config, dataset, axis='xz', save_figure_path='', figname=None, keys=keys, frames=frames, ax=ax_xz, save=False)
    plot_heatmap(config, dataset, axis='yz', save_figure_path='', figname=None, keys=keys, frames=frames, ax=ax_yz, save=False)
    
    xyz_axis_grid.set_spines_and_labels(ax_xy, ax_xz, ax_yz)
    
    if title is not None:
        plt.suptitle(title)

    if save:
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
        
    n_frames = 0
    for key in keys:
        trajec = dataset.trajecs[key]
        n_frames += trajec.length
    
    binres = 0.003
    binsx = int(1.2/binres)
    binsy = int(0.3/binres)
    binsz = int(0.3/binres)
    
    if axis == 'xy' or axis == 'xz':
        depth = 0.3
    if axis == 'yz':
        depth = 1.2
    colornorm = [0,n_frames/2500.*depth]
    print 'Colornorm: ', colornorm
    
    if axis == 'xy':
        depth_range = [-.18, 0.05]
        print depth_range
    elif axis == 'xz': 
        depth_range = None
    elif axis == 'yz': 
        depth_range = None
        
    fap.heatmap(ax, dataset, axis=axis, keys=keys, xticks=config.ticks['x'], yticks=config.ticks['y'], zticks=config.ticks['z'], rticks=config.ticks['r'], colornorm=colornorm, normalize_for_speed=False, bins=[binsx,binsy,binsz], depth_range=depth_range)
    
    
    height = config.post_center[2]-config.ticks['z'][0]
    if config.post:
        postcolor = 'black'
    else:
        postcolor = 'green' 
    
    artists = []
    if axis == 'xy':
        ax.set_xlabel('x axis, upwind negative')
        ax.set_ylabel('y axis')
        post = patches.Circle(config.post_center[0:2], config.post_radius, color=postcolor, edgecolor='none')
        artists.append(post)
    if axis == 'xz':
        ax.set_xlabel('x axis, upwind negative')
        ax.set_ylabel('z axis')
        post = patches.Rectangle([-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color=postcolor, edgecolor='none')
        artists.append(post)
    if axis == 'yz':
        ax.set_xlabel('y axis')
        ax.set_ylabel('z axis')
        post = patches.Rectangle([-1*config.post_radius+config.post_center[1], config.ticks['z'][0]], config.post_radius*2, height, color=postcolor, edgecolor='none')
        artists.append(post)
    if axis == 'rz':
        ax.set_xlabel('r axis')
        ax.set_ylabel('z axis')
        post = patches.Rectangle([0.0,config.ticks['z'][0]], config.post_radius, height, color='black')
        artists.append(post)
    
    if axis == 'xy':
        for i, circle_position in enumerate(config.dots_on_floor_position):
            radius = config.dots_on_floor_radius[i]
            circle = patches.Circle(circle_position, radius, facecolor='none', edgecolor='white', linewidth=2)
            ax.add_artist(circle)
            
        radius = config.dots_on_wall_radius[0]
        rect = patches.Rectangle( (circle_position[0]-radius, -.14), radius*2, 0.01, facecolor='none', edgecolor='white', linewidth=2)
        ax.add_artist(rect)
        
        rect = patches.Rectangle( (circle_position[0]-radius, .14), radius*2, 0.01, facecolor='none', edgecolor='white', linewidth=2)
        ax.add_artist(rect)
            
    if axis == 'xz':
        for i, circle_position in enumerate(config.dots_on_wall_position):
            radius = config.dots_on_wall_radius[i]
            circle = patches.Circle(circle_position, radius, facecolor='none', edgecolor='white', linewidth=2)
            ax.add_artist(circle)
            
        for i, circle_position in enumerate(config.dots_on_floor_position):
            radius = config.dots_on_floor_radius[i]
            rect = patches.Rectangle( (circle_position[0]-radius, -.14), radius*2, 0.01, facecolor='none', edgecolor='white', linewidth=2)
            ax.add_artist(rect)
        
    if 0:
        if artists is not None:
            for artist in artists:
                ax.add_artist(artist)
        
    if save:
        if figname is None:
            figname = save_figure_path + 'heatmap_' + axis + '.pdf'
        else:
            figname = save_figure_path + figname
        fig.savefig(figname, format='pdf')
    

def pdf_book(config, dataset, save_figure_path=''):
    if save_figure_path == '':
        figure_path = os.path.join(config.path, config.figure_path)
        save_figure_path=os.path.join(figure_path, 'heatmaps/')

    threshold_odor=10
        
    for odor in [True, False]:
        book_name = 'heatmap_book_odor_' + str(odor) + '.pdf'
        pdf_name_with_path = os.path.join(save_figure_path, book_name)
        pp = PdfPages(pdf_name_with_path)
        
        print
        print 'Odor Stimuli: ', config.odor_stimulus.keys()
        
        key_set = {}
        for odor_stimulus in config.odor_stimulus.keys():
            
            keys_tmp = fad.get_keys_with_attr(dataset, 'odor_stimulus', odor_stimulus)
            keys = []
            for key in keys_tmp:
                trajec = dataset.trajecs[key]
                if odor is True:
                    if np.max(trajec.odor) > threshold_odor:
                        keys.append(key)
                elif odor is False:
                    if np.max(trajec.odor) < threshold_odor:
                        keys.append(key)
            
            if len(keys) > 0:    
                key_set.setdefault(odor_stimulus, keys)

        for odor_stimulus, keys in key_set.items():

            fig = plt.figure()
            figname = 'heatmap_' + odor_stimulus + '_odor_all_axes.pdf'
            title = 'odor ' + odor_stimulus 
            
            plot_all_heatmaps(config, dataset, save_figure_path=save_figure_path, figname=figname, title=title, keys=keys)
            pp.savefig()
            plt.close('all')
            
        pp.close()
    
    
    
if __name__ == '__main__':
    config = analysis_configuration.Config()
    culled_dataset = fad.load('../' + config.culled_datasets_path + config.culled_dataset_name)
    
    pdf_book(config, culled_dataset, save_figure_path='../../figures/heatmaps/')
    


