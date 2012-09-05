import os, sys
sys.path.append('../analysis_modules')
import fly_plot_lib
fly_plot_lib.set_params.pdf()
from matplotlib.backends.backend_pdf import PdfPages

import fly_plot_lib.plot as fpl
import odor_packet_analysis as opa

import flydra_analysis_tools.trajectory_analysis_core as tac

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import xyz_axis_grid

def plot_odor_traces_book(path, config, dataset, keys=None, show_saccades=False, frames_to_show_before_odor='all', frames_to_show_after_odor='all', odor_multiplier=1):


    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'odor_trace_book.pdf')
    pp = PdfPages(pdf_name_with_path)

    for key in keys:
        print key
        plot_odor_traces(path, config, dataset, keys=[key], show_saccades=show_saccades, frames_to_show_before_odor=frames_to_show_before_odor, frames_to_show_after_odor=frames_to_show_after_odor, odor_multiplier=odor_multiplier)
        pp.savefig()
        plt.close('all')
        
    pp.close()

def plot_odor_traces(path, config, dataset, keys=None, show_saccades=False, frames_to_show_before_odor='all', frames_to_show_after_odor='all', save=False, odor_multiplier=1):

    ax_xy, ax_xz, ax_yz = xyz_axis_grid.get_axes()
    
    plot_odor_trace_on_ax(path, config, dataset, keys=keys, axis='xy', show_saccades=show_saccades, frames_to_show_before_odor=frames_to_show_before_odor, frames_to_show_after_odor=frames_to_show_after_odor, ax=ax_xy, odor_multiplier=odor_multiplier)
    
    plot_odor_trace_on_ax(path, config, dataset, keys=keys, axis='xz', show_saccades=show_saccades, frames_to_show_before_odor=frames_to_show_before_odor, frames_to_show_after_odor=frames_to_show_after_odor, ax=ax_xz, odor_multiplier=odor_multiplier)
    
    plot_odor_trace_on_ax(path, config, dataset, keys=keys, axis='yz', show_saccades=show_saccades, frames_to_show_before_odor=frames_to_show_before_odor, frames_to_show_after_odor=frames_to_show_after_odor, ax=ax_yz, odor_multiplier=odor_multiplier)
    
    xyz_axis_grid.set_spines_and_labels(ax_xy, ax_xz, ax_yz)
    
    if save:
        figname = None
        figure_path = os.path.join(path, config.figure_path)
        save_figure_path = os.path.join(figure_path, 'odor_traces/')
        
        if figname is None:
            figname = save_figure_path + 'odor_trace' + '.pdf'
        else:
            figname = os.path.join(save_figure_path, figname)
        plt.savefig(figname, format='pdf')


def plot_odor_trace_on_ax(path, config, dataset, keys=None, axis='xy', show_saccades=False, frames_to_show_before_odor='all', frames_to_show_after_odor='all', ax=None, odor_multiplier=1):
    # test with '0_9174'
    
    if keys is None: 
        keys = dataset.trajecs.keys()

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    colormap='jet'
    linewidth = 1
    alpha = 1
    zorder = 1
    norm = (0,100)
    show_start = False
    color_attribute = 'odor'
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
        
        frames_where_odor = np.where(trajec.odor > 10)[0]
        
        if frames_to_show_before_odor == 'all':
            frame0 = 0
        else:
            frame0 = np.min(frames_where_odor) - frames_to_show_before_odor
            frame0 = np.max([frame0, 0])
        if frames_to_show_after_odor == 'all':
            frame1 = trajec.length
        else:
            frame1 = np.argmax(trajec.odor) + frames_to_show_after_odor
            frame1 = np.min([trajec.length, frame1])
        frames = np.arange(frame0, frame1)
        
        tac.calc_heading_for_axes(trajec, axis=axis)
        orientation = trajec.__getattribute__('heading_smooth_'+axis)
        
        fpl.colorline_with_heading(ax,trajec.positions[frames,axes[0]], trajec.positions[frames,axes[1]], c[frames]*odor_multiplier, orientation=orientation[frames], colormap=colormap, alpha=alpha, colornorm=norm, size_radius=trajec.speed[frames], size_radius_range=[0.005, .02], deg=False, nskip=2, center_point_size=0.01)
        
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
    
    
def plot_odor_traces_for_odor_puffs_before_post(path, config, dataset, axis='xy'):
    keys = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=50, threshold_distance=0.1)
    plot_odor_traces(path, config, dataset, keys=keys, axis=axis)
    









