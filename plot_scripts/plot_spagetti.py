#!/usr/bin/env python
import sys, os
from optparse import OptionParser
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

import odor_packet_analysis as opa

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

################# get trajectory keys #################

def get_keys(dataset):
    keys = dataset.trajecs.keys()
    return keys

################# plotting functions #######################

def plot_colored_cartesian_spagetti(config, dataset, axis='xy', xlim=(-0.2, .2), ylim=(-0.75, .25), zlim=(0, 0.3), keys=None, keys_to_highlight=[], show_saccades=False, colormap='jet', color_attribute='speed', norm=(0,0.5), artists=None, save_figure_path='', figname=None, show_start=False):
    if keys is None:
        keys = get_keys(dataset)
    print 'plotting spagetti, axis: ', axis
    print 'number of keys: ', len(keys)
    if len(keys) < 1:
        print 'No data'
        return
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    height = config.post_center[2]-config.ticks['z'][0]
    
    print 'ARTISTS STARTING'
    print artists
    
    if axis=='xy': # xy plane
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_autoscale_on(True)
        ax.set_aspect('equal')
        axes=[0,1]
        fap.cartesian_spagetti(ax, dataset, keys=keys, nkeys=10, start_key=0, axes=axes, show_saccades=show_saccades, keys_to_highlight=[], colormap=colormap, color_attribute=color_attribute, norm=norm, show_start=show_start)
        post = patches.Circle(config.post_center[0:2], config.post_radius, color='black')
        
    if axis=='yz': # yz plane
        ax.set_ylim(zlim[0], zlim[1])
        ax.set_xlim(ylim[0], ylim[1])
        ax.set_autoscale_on(True)
        ax.set_aspect('equal')
        axes=[1,2]
        fap.cartesian_spagetti(ax, dataset, keys=keys, nkeys=10, start_key=0, axes=axes, show_saccades=show_saccades, keys_to_highlight=[], colormap=colormap, color_attribute=color_attribute, norm=norm, show_start=show_start)
        post = patches.Rectangle([-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black')
        
    if axis=='xz': # xz plane
        ax.set_ylim(zlim[0], zlim[1])
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_autoscale_on(True)
        ax.set_aspect('equal')
        axes=[0,2]
        fap.cartesian_spagetti(ax, dataset, keys=keys, nkeys=10, start_key=0, axes=axes, show_saccades=show_saccades, keys_to_highlight=[], colormap=colormap, color_attribute=color_attribute, norm=norm, show_start=show_start)
        post = patches.Rectangle([-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black')
        
    if artists is None:
        artists = []
    artists.append(post)
    if artists is not None:
        for artist in artists:
            ax.add_artist(artist)

    #prep_cartesian_spagetti_for_saving(ax)
    xticks = config.ticks['x']
    yticks = config.ticks['y']
    zticks = config.ticks['z']
    
    if axis=='xy':
        fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks)
        ax.set_xlabel('x axis, m')
        ax.set_ylabel('y axis, m')
        ax.set_title('xy plot, color=speed from 0-0.5 m/s')

    if axis=='yz':
        fpl.adjust_spines(ax, ['left', 'bottom'], xticks=yticks, yticks=zticks)
        ax.set_xlabel('y axis, m')
        ax.set_ylabel('z axis, m')
        ax.set_title('yz plot, color=speed from 0-0.5 m/s')
        
    if axis=='xz':
        fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=zticks)
        ax.set_xlabel('x axis, m')
        ax.set_ylabel('z axis, m')
        ax.set_title('xz plot, color=speed from 0-0.5 m/s')
        
    fig.set_size_inches(8,8)
    if figname is None:
        figname = save_figure_path + 'spagetti_' + axis + '.pdf'
    else:
        figname = os.path.join(save_figure_path, figname)
    fig.savefig(figname, format='pdf')

    return ax
    
    
    
def main(config, culled_dataset, save_figure_path=''):
    print
    print 'Plotting spagetti'
    
    if 1:
        # in odor
        print
        print 'Odor: '
        dataset_in_odor = fad.make_dataset_with_attribute_filter(culled_dataset, 'odor_stimulus', 'on')
        if len(dataset_in_odor.trajecs.keys()) > 0:
            plot_colored_cartesian_spagetti(config, dataset_in_odor, axis='xy', save_figure_path=save_figure_path, figname='spagetti_odor_xy.pdf')
            plot_colored_cartesian_spagetti(config, dataset_in_odor, axis='yz', save_figure_path=save_figure_path, figname='spagetti_odor_yz.pdf')
            plot_colored_cartesian_spagetti(config, dataset_in_odor, axis='xz', save_figure_path=save_figure_path, figname='spagetti_odor_xz.pdf')

        # not in odor
        print
        print 'No odor: '
        dataset_no_odor = fad.make_dataset_with_attribute_filter(culled_dataset, 'odor_stimulus', 'none')
        if len(dataset_no_odor.trajecs.keys()) > 0:
            plot_colored_cartesian_spagetti(config, dataset_no_odor, axis='xy', save_figure_path=save_figure_path, figname='spagetti_no_odor_xy.pdf')
            plot_colored_cartesian_spagetti(config, dataset_no_odor, axis='yz', save_figure_path=save_figure_path, figname='spagetti_no_odor_yz.pdf')
            plot_colored_cartesian_spagetti(config, dataset_no_odor, axis='xz', save_figure_path=save_figure_path, figname='spagetti_no_odor_xz.pdf')
        
        # pulse odor
        print
        print 'Pulsing odor: '
        dataset_pulsing_odor = fad.make_dataset_with_attribute_filter(culled_dataset, 'odor_stimulus', 'pulsing')
        if len(dataset_pulsing_odor.trajecs.keys()) > 0:
            plot_colored_cartesian_spagetti(config, dataset_pulsing_odor, axis='xy', save_figure_path=save_figure_path, figname='spagetti_pulsing_odor_xy.pdf')
            plot_colored_cartesian_spagetti(config, dataset_pulsing_odor, axis='yz', save_figure_path=save_figure_path, figname='spagetti_pulsing_odor_yz.pdf')
            plot_colored_cartesian_spagetti(config, dataset_pulsing_odor, axis='xz', save_figure_path=save_figure_path, figname='spagetti_pulsing_xz.pdf')
            
        
    # odor plot
    print
    print 'Best odor trajectory: '
    if 1:
        keys = opa.get_trajectories_with_odor(culled_dataset, 50)
        keys = keys[0]
        plot_colored_cartesian_spagetti(config, culled_dataset, axis='xy', keys=keys, color_attribute='odor', norm=(0,200), save_figure_path=save_figure_path, figname='odor_trajectory_xy.pdf', show_start=True)
        plot_colored_cartesian_spagetti(config, culled_dataset, axis='xz', keys=keys, color_attribute='odor', norm=(0,200), save_figure_path=save_figure_path, figname='odor_trajectory_xz.pdf', show_start=True)
    
    if 0:
        keys = opa.get_trajectories_with_odor(culled_dataset, 175)
        plot_colored_cartesian_spagetti(config, culled_dataset, axis='xy', keys=keys, color_attribute='odor', norm=(0,100), save_figure_path=save_figure_path, figname='odor_trajectory_xy.pdf')
        plot_colored_cartesian_spagetti(config, culled_dataset, axis='xz', keys=keys, color_attribute='odor', norm=(0,100), save_figure_path=save_figure_path, figname='odor_trajectory_xz.pdf')
    

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to empty data folder, where you have a configuration file")
    (options, args) = parser.parse_args()
    
    path = options.path    
    sys.path.append(path)
    import analysis_configuration
    config = analysis_configuration.Config()
    
    
    culled_dataset_name = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name)
    culled_dataset = fad.load(culled_dataset_name)
    
    figure_path = os.path.join(path, config.figure_path)
    main(config, culled_dataset, save_figure_path=os.path.join(figure_path, 'spagetti/') )
    


