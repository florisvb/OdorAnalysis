import os, sys
sys.path.append('../analysis_modules')
import imp
from optparse import OptionParser

import fly_plot_lib
fly_plot_lib.set_params.pdf()
from matplotlib.backends.backend_pdf import PdfPages

import fly_plot_lib.plot as fpl
import odor_packet_analysis as opa

import flydra_analysis_tools.trajectory_analysis_core as tac
import flydra_analysis_tools.flydra_analysis_dataset as fad
import help_functions as hf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import xyz_axis_grid


def print_trace_end_stats(dataset):
    keys = dataset.trajecs.keys()
    for key in keys:
        trajec = dataset.trajecs[key]
        
        if trajec.post_behavior != 'landing':
            if trajec.odor_stimulus != 'none':
                if np.max(trajec.distance_to_post) > 0.03:
                    if trajec.distance_to_post[-1] < 0.01:
                        print key, 'speed: ', trajec.speed[-1], np.max(trajec.speed), 'distance: ', trajec.distance_to_post[-1], np.max(trajec.distance_to_post), 'len: ', trajec.length, trajec.post_behavior

def plot_post_traces(path, config, dataset, keys=None, save=True):
    if keys is None:
        keys = dataset.trajecs.keys()
        
    ax_xy, ax_xz, ax_yz = xyz_axis_grid.get_axes()
    
    for key in keys:
        trajec = dataset.trajecs[key]
        
        if trajec.post_behavior != 'landing':
            if trajec.odor_stimulus != 'none':
                if trajec.distance_to_post[-1] < 0.01:
                
                    print key
                    plot_odor_trace_on_ax(path, config, dataset, keys=[key], axis='xy', ax=ax_xy, show_post=config.post)
                    plot_odor_trace_on_ax(path, config, dataset, keys=[key], axis='xz', ax=ax_xz, show_post=config.post)
                    plot_odor_trace_on_ax(path, config, dataset, keys=[key], axis='yz', ax=ax_yz, show_post=config.post)
                    
    xyz_axis_grid.set_spines_and_labels(ax_xy, ax_xz, ax_yz)
    
    if save:
        figname = None
        figure_path = os.path.join(path, config.figure_path)
        save_figure_path = os.path.join(figure_path, 'odor_traces/')
        
        if figname is None:
            figname = save_figure_path + 'trajec_ending_near_post' + '.pdf'
        else:
            figname = os.path.join(save_figure_path, figname)
        plt.savefig(figname, format='pdf')
        
        


def plot_trace_ends(path, config, dataset, keys=None, show_saccades=False, save=True):
    if keys is None:
        keys = dataset.trajecs.keys()
        
    ax_xy, ax_xz, ax_yz = xyz_axis_grid.get_axes()
    
    if config.post:
        height = config.post_center[2]-config.ticks['z'][0]
        post = patches.Circle(config.post_center[0:2], config.post_radius, color='black')
        ax_xy.add_artist(post)
        post = patches.Rectangle([config.post_center[0]-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black')
        ax_xz.add_artist(post)
        post = patches.Rectangle([config.post_center[1]-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black')
        ax_yz.add_artist(post)    
    
    for key in keys:
        trajec = dataset.trajecs[key]
        
        if trajec.post_behavior == 'landing':
            color = 'green'
        else:
            color = 'red'
        
        if np.max(trajec.distance_to_post) > 0.05:
        
            
            if trajec.positions[-1][2] < 0.0:
                ax_xy.plot(trajec.positions[-1][0],trajec.positions[-1][1],'.', color=color, markersize=1, alpha=0.7)
            
            if np.abs(trajec.positions[-1][1]) < 0.03:
                ax_xz.plot(trajec.positions[-1][0],trajec.positions[-1][2],'.', color=color, markersize=1, alpha=0.7)
                
            if np.abs(trajec.positions[-1][0]) < 0.05:
                ax_yz.plot(trajec.positions[-1][1],trajec.positions[-1][2],'.', color=color, markersize=1, alpha=0.7)

    
    xyz_axis_grid.set_spines_and_labels(ax_xy, ax_xz, ax_yz)
    
    if save:
        figname = None
        figure_path = os.path.join(path, config.figure_path)
        save_figure_path = os.path.join(figure_path, 'odor_traces/')
        
        if figname is None:
            figname = save_figure_path + 'trajec_ends' + '.pdf'
        else:
            figname = os.path.join(save_figure_path, figname)
        plt.savefig(figname, format='pdf')
        
        
        

def plot_odor_traces_book(path, config, dataset, keys=None, show_saccades=True, frames_to_show_before_odor='all', frames_to_show_after_odor='all', odor_multiplier=1, book_name='odor_trace_book.pdf'):


    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, book_name)
    pp = PdfPages(pdf_name_with_path)

    for key in keys:
        print key
        plot_odor_traces(path, config, dataset, keys=[key], show_saccades=show_saccades, frames_to_show_before_odor=frames_to_show_before_odor, frames_to_show_after_odor=frames_to_show_after_odor, odor_multiplier=odor_multiplier)
        
        title_text = key.replace('_', '-')
        trajec = dataset.trajecs[key]
        if 0:
            behavior_text = ''
            for behavior in trajec.post_behavior:
                behavior_text += behavior + ', '
            if len(behavior_text) > 0:
                title_text += ' -- behavior: ' + behavior_text
            
            title_text += 'visual: ' + trajec.visual_stimulus
            
        plt.suptitle(title_text.strip())
        pp.savefig()
        plt.close('all')
        
    pp.close()
    
    

def plot_odor_traces(path, config, dataset, keys=None, show_saccades=False, frames_to_show_before_odor='all', frames_to_show_after_odor='all', save=False, odor_multiplier=1, frameranges=None):

    ax_xy, ax_xz, ax_yz = xyz_axis_grid.get_axes()
    
    plot_odor_trace_on_ax(path, config, dataset, keys=keys, axis='xy', show_saccades=show_saccades, frames_to_show_before_odor=frames_to_show_before_odor, frames_to_show_after_odor=frames_to_show_after_odor, ax=ax_xy, odor_multiplier=odor_multiplier, show_post=config.post, frameranges=frameranges)
    
    plot_odor_trace_on_ax(path, config, dataset, keys=keys, axis='xz', show_saccades=show_saccades, frames_to_show_before_odor=frames_to_show_before_odor, frames_to_show_after_odor=frames_to_show_after_odor, ax=ax_xz, odor_multiplier=odor_multiplier, show_post=config.post, frameranges=frameranges)
    
    plot_odor_trace_on_ax(path, config, dataset, keys=keys, axis='yz', show_saccades=show_saccades, frames_to_show_before_odor=frames_to_show_before_odor, frames_to_show_after_odor=frames_to_show_after_odor, ax=ax_yz, odor_multiplier=odor_multiplier, show_post=config.post, frameranges=frameranges)
    
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


def plot_odor_trace_on_ax(path, config, dataset, keys=None, axis='xy', show_saccades=False, frames_to_show_before_odor='all', frames_to_show_after_odor='all', ax=None, odor_multiplier=1, show_post=True, save=False, frameranges=None):
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
    
    postx = config.post_center[0]
    posty = config.post_center[1]
    postz = config.post_center[2]
    
    if axis == 'xy':
        axes=[0,1]
        post = patches.Circle(config.post_center[0:2], config.post_radius, color='black')
        depth = 2
    if axis == 'xz':
        axes=[0,2]
        post = patches.Rectangle([postx-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black')
        depth = 1
    if axis == 'yz':
        axes=[1,2]
        post = patches.Rectangle([posty-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black')
        depth = 0
    
    ##########################
    # Plot trajectory
    ##########################
    
    for key in keys:
        trajec = dataset.trajecs[key]
        c = trajec.__getattribute__(color_attribute)
        
        frames_where_odor = np.where(trajec.odor > 10)[0]
        #frames_where_odor = hf.find_continuous_blocks(frames_where_odor, 5, return_longest_only=True)
        
        if frameranges is not None:
            if frameranges.has_key(key):
                frames = np.arange(frameranges[key][0], frameranges[key][-1])
                autoframerange = False
            else:
                autoframerange = True
        else:
            autoframerange = True
        if autoframerange:
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
        
        fpl.colorline_with_heading(ax,trajec.positions[frames,axes[0]], trajec.positions[frames,axes[1]], c[frames]*odor_multiplier, orientation=orientation[frames], colormap=colormap, alpha=alpha, colornorm=norm, size_radius=0.15-np.abs(trajec.positions[frames,depth]), size_radius_range=[0.003, .025], deg=False, nskip=2, center_point_size=0.01)
        #fpl.colorline(ax,trajec.positions[frames,axes[0]], trajec.positions[frames,axes[1]], c[frames]*odor_multiplier, colormap=colormap, alpha=alpha)
        
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
    if show_post:
        artists.append(post)
    if artists is not None:
        for artist in artists:
            ax.add_artist(artist)
            
    if save:
        ax.set_xlim(-.4, 1.)
        ax.set_ylim(-.2,.2)
        fig.savefig('odor_traces.pdf', format='pdf')
    
    
def plot_odor_traces_for_odor_puffs_before_post(path, config, dataset, axis='xy'):
    keys = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=50, threshold_distance=0.1)
    plot_odor_traces(path, config, dataset, keys=keys, axis=axis)
    




def pdf_book(config, dataset, save_figure_path=''):
    if save_figure_path == '':
        figure_path = os.path.join(config.path, config.figure_path)
        save_figure_path=os.path.join(figure_path, 'odor_traces/')

    threshold_odor=10
        
    for odor in [True, False]:
        
        key_set = {}
        for odor_stimulus in config.odor_stimulus.keys():
            #keys_tmp = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=threshold_odor, odor_stimulus=odor_stimulus, threshold_distance_min=0.1, odor=odor)
            keys_tmp = fad.get_keys_with_attr(dataset, 'odor_stimulus', odor_stimulus)
            
            keys = []
            for key in keys_tmp:
                trajec = dataset.trajecs[key]
                add_key = True
                if trajec.positions[0,0] < 0.3:
                    add_key = False
                if odor:
                    frames_in_odor = np.where(trajec.odor > threshold_odor)[0]
                    if len(frames_in_odor) < 20:
                        add_key = False
                if add_key:
                    keys.append(key)
            
            print odor_stimulus, keys
            if len(keys) > 0:    
                key_set.setdefault(odor_stimulus, keys)

        for odor_stimulus, keys in key_set.items():
            print 'Odor Book, Chapter: ', odor_stimulus
            
            if len(keys) < 250:
                keys_to_plot = keys
            else:
                keys_to_plot = keys[250:500]
                
            book_name = 'odor_trace_book_' + odor_stimulus + '_' + str(odor) + '.pdf'
                
            plot_odor_traces_book(config.path, config, dataset, keys=keys_to_plot, show_saccades=False, frames_to_show_before_odor=100, frames_to_show_after_odor=100, odor_multiplier=1, book_name=book_name)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to data folder, where you have a configuration file")
                        
    (options, args) = parser.parse_args()
    
    path = options.path    
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
    dataset = fad.load(culled_dataset_filename)

    pdf_book(config, dataset, save_figure_path='')



