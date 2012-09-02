import os, sys
sys.path.append('../analysis_modules')
import fly_plot_lib
fly_plot_lib.set_params.pdf()
import fly_plot_lib.plot as fpl
import odor_packet_analysis as opa

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def plot_odor_traces(path, config, dataset, keys=None, axis='xy', show_saccades=False, frames_to_show_before_odor='all'):
    # test with '0_9174'
    
    if keys is None: 
        keys = dataset.trajecs.keys()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    colormap='jet'
    linewidth = 1
    alpha = 1
    zorder = 1
    norm = (0,100)
    show_start = False
    color_attribute = 'odor'
    figname = None
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    artists = None
    height = config.post_center[2]-config.ticks['z'][0]
    
    if axis == 'xy':
        axes=[0,1]
        post = patches.Circle(config.post_center[0:2], config.post_radius, color='black')
    if axis == 'xz':
        axes=[0,2]
        post = patches.Rectangle([-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black')
    if axis == 'yz':
        axes=[1,2]
        post = patches.Rectangle([-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black')
    
    ##########################
    # Plot trajectory
    ##########################
    
    for key in keys:
        trajec = dataset.trajecs[key]
        c = trajec.__getattribute__(color_attribute)
        
        if frames_to_show_before_odor == 'all':
            frame0 = 0
        else:
            frame0 = np.argmax(trajec.odor) - frames_to_show_before_odor
        frames = np.arange(frame0, trajec.length)
        
        fpl.colorline_with_heading(ax,trajec.positions[frames,axes[0]], trajec.positions[frames,axes[1]], c[frames], orientation=trajec.heading_smooth[frames], colormap=colormap, alpha=alpha, colornorm=norm, size_radius=trajec.speed[frames], size_radius_range=[0.001, .01], deg=False, nskip=2, center_point_size=0.01)
        
        if show_start:
            start = patches.Circle( (trajec.positions[frames[0],axes[0]], trajec.positions[frames[0],axes[1]]), radius=0.004, facecolor='green', edgecolor='none', linewidth=0, alpha=1, zorder=zorder+1)
            ax.add_artist(start)
        
        if show_saccades:
            for sac_range in trajec.saccades:
                if sac_range[0] in frames and sac_range[-1] in frames: 
                    middle_saccade_index = int(len(sac_range)/2.)
                    middle_saccade_frame = sac_range[middle_saccade_index]
                    saccade = patches.Circle( (trajec.positions[middle_saccade_frame,axes[0]], trajec.positions[middle_saccade_frame,axes[1]]), radius=0.004, facecolor='red', edgecolor='none', linewidth=0, alpha=1, zorder=zorder+1)
                    ax.add_artist(saccade)
            
        
        
    ############################
    # Add post, make plot pretty
    ############################
    
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
    
    ax.set_aspect('equal')
    
    if axis=='xy':
        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_ylim(yticks[0], yticks[-1])
        fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks)
        ax.set_xlabel('x axis, m')
        ax.set_ylabel('y axis, m')
        ax.set_title('xy plot, color=odor from 0-100')

    if axis=='yz':
        ax.set_xlim(yticks[0], yticks[-1])
        ax.set_ylim(zticks[0], zticks[-1])
        fpl.adjust_spines(ax, ['left', 'bottom'], xticks=yticks, yticks=zticks)
        ax.set_xlabel('y axis, m')
        ax.set_ylabel('z axis, m')
        ax.set_title('yz plot, color=odor from 0-100')
        
    if axis=='xz':
        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_ylim(zticks[0], zticks[-1])
        fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=zticks)
        ax.set_xlabel('x axis, m')
        ax.set_ylabel('z axis, m')
        ax.set_title('xz plot, color=odor from 0-100')
    
    fig.set_size_inches(8,4)
    
    if figname is None:
        figname = save_figure_path + 'odor_trace_' + axis + '.pdf'
    else:
        figname = os.path.join(save_figure_path, figname)
    fig.savefig(figname, format='pdf')
    
    
def plot_odor_traces_for_odor_puffs_before_post(path, config, dataset, axis='xy'):
    keys = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=50, threshold_distance=0.1)
    plot_odor_traces(path, config, dataset, keys=keys, axis=axis)
    









