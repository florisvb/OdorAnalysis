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
    
    fap.heatmap(ax, dataset, axis=axis, keys=keys, xticks=config.ticks['x'], yticks=config.ticks['y'], zticks=config.ticks['z'], rticks=config.ticks['r'], colornorm=colornorm, normalize_for_speed=False, frame_list=frames, bins=[binsx,binsy,binsz])
    
    
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
            #keys = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=threshold_odor, odor_stimulus=odor_stimulus, threshold_distance_min=100, odor=odor)
            
            keys = fad.get_keys_with_attr(dataset, 'odor_stimulus', odor_stimulus)
            
            if len(keys) > 0:    
                key_set.setdefault(odor_stimulus, keys)

        for odor_stimulus, keys in key_set.items():

            '''

            print 'Odor Book, Chapter: ', odor_stimulus
            landers = fad.get_keys_with_attr(dataset, ['post_behavior'], ['landing'], keys=keys)
            boomerangs = fad.get_keys_with_attr(dataset, ['post_behavior'], ['boomerang'], keys=keys)
            landing_ratio = float(len(landers)) / float(len(keys))
            
            
            # count flies that stayed for less than 0.1 sec, and were observed leaving the post:
            n_stayed_less_than_025 = 0
            for key in keys:
                trajec = dataset.trajecs[key]
                if trajec.residency_time is not None:
                    if trajec.residency_time < 0.1 and 'boomerang' in trajec.post_behavior:
                        n_stayed_less_than_025 += 1
            
            # count flies that stayed for 0.1 sec and were not observed after that
            n_stayed_less_than_025_dissappeared = 0
            for key in keys:
                trajec = dataset.trajecs[key]
                if trajec.residency_time is not None:
                    if trajec.residency_time < 0.1 and 'boomerang' not in trajec.post_behavior:
                        n_stayed_less_than_025_dissappeared += 1
                        
            # control: flies that stayed more than .1 sec
            n_stayed_more_than_025 = 0
            for key in keys:
                trajec = dataset.trajecs[key]
                if trajec.residency_time is not None:
                    if trajec.residency_time > 0.1:
                        n_stayed_more_than_025 += 1   
                        
            # flies that stayed for more than 1 sec   
            n_stayed_more_than_100 = 0
            for key in keys:
                trajec = dataset.trajecs[key]
                if trajec.residency_time is not None:
                    if trajec.residency_time > 1.00:
                        n_stayed_more_than_100 += 1   
                        
            # count flies that stayed for less than 1.0 sec, and were observed leaving the post:
            n_stayed_less_than_100 = 0
            for key in keys:
                trajec = dataset.trajecs[key]
                if trajec.residency_time is not None:
                    if trajec.residency_time < 1. and 'boomerang' in trajec.post_behavior:
                        n_stayed_less_than_100 += 1
                        
            # count flies that stayed for less than 1.0 sec, and were not observered after that:
            n_stayed_less_than_100_dissappeared = 0
            for key in keys:
                trajec = dataset.trajecs[key]
                if trajec.residency_time is not None:
                    if trajec.residency_time < 1. and 'boomerang' not in trajec.post_behavior:
                        n_stayed_less_than_100_dissappeared += 1
                        
            # flies that stayed for more than 5 sec   
            n_stayed_more_than_500 = 0
            for key in keys:
                trajec = dataset.trajecs[key]
                if trajec.residency_time is not None:
                    if trajec.residency_time > 5.00:
                        n_stayed_more_than_500 += 1   
                        
            # count flies that stayed for less than 5 sec, and were observed leaving the post:
            n_stayed_less_than_500 = 0
            for key in keys:
                trajec = dataset.trajecs[key]
                if trajec.residency_time is not None:
                    if trajec.residency_time < 5. and 'boomerang' in trajec.post_behavior:
                        n_stayed_less_than_500 += 1
                        
            # count flies that stayed for less than 5 sec, and were not observered after that:
            n_stayed_less_than_500_dissappeared = 0
            for key in keys:
                trajec = dataset.trajecs[key]
                if trajec.residency_time is not None:
                    if trajec.residency_time < 5. and 'boomerang' not in trajec.post_behavior:
                        n_stayed_less_than_500_dissappeared += 1
            
                        
            if n_stayed_less_than_025 > 0:
                stayed_less_than_025_lower = n_stayed_less_than_025 / float(n_stayed_less_than_025 + n_stayed_more_than_025)
                stayed_less_than_025_upper = (n_stayed_less_than_025 + n_stayed_less_than_025_dissappeared) / float(n_stayed_less_than_025 + n_stayed_more_than_025)
            else:
                stayed_less_than_025_lower = 0
                stayed_less_than_025_upper = 0
            
            if n_stayed_more_than_100 > 0: 
                stayed_more_than_100_upper = n_stayed_more_than_100 / float(n_stayed_less_than_100 + n_stayed_more_than_100)
                stayed_more_than_100_lower = n_stayed_more_than_100 / float(n_stayed_less_than_100 + n_stayed_less_than_100_dissappeared + n_stayed_more_than_100)
            else:
                stayed_more_than_100_upper = 0
                stayed_more_than_100_lower = 0
            
            if n_stayed_more_than_500 > 0:
                stayed_more_than_500_upper = n_stayed_more_than_500 / float(n_stayed_less_than_500 + n_stayed_more_than_500)
                stayed_more_than_500_lower = n_stayed_more_than_500 / float(n_stayed_less_than_500 + n_stayed_less_than_500_dissappeared + n_stayed_more_than_500)
            else:
                stayed_more_than_500_upper = 0
                stayed_more_than_500_lower = 0
            
            trajec_lengths = []
            for key in keys:
                trajec = dataset.trajecs[key]
                trajec_lengths.append(trajec.length/trajec.fps)
            
            '''
            fig = plt.figure()
            figname = 'heatmap_' + odor_stimulus + '_odor_all_axes.pdf'
            title = 'odor ' + odor_stimulus 
            '''
            if odor:
                title += ' - All trajectories came within 10 cm of the post, and all experienced odor'
            else:
                title += ' - All trajectories came within 10 cm of the post, NONE experienced odor'
            
            title += '\n' + 'nflies: ' + str(len(keys)) + ' -- landings: ' + str(len(landers)) + ' -- landing percentage: ' + str(landing_ratio*100)[0:2] + '\%' 
            
            title += ' -- ' + str(len(boomerangs)) + ' observed leaving post as well.'
            
            title += '\n' + 'Of the landings, ' + str(stayed_less_than_025_lower*100)[0:3] + '-' + str(stayed_less_than_025_upper*100)[0:3] + '\%' + ' only stayed on the post for 0.1 sec or less'# + ' N: ' + str(n_stayed_less_than_025) + ', ' + str(n_stayed_less_than_025 + n_stayed_more_than_025)
            
            title += '\n' + 'Of the landings, ' + str(stayed_more_than_100_lower*100)[0:3] + '-' + str(stayed_more_than_100_upper*100)[0:3] + '\%' + ' stayed on the post for 1.0 sec or more'# + ' N: ' + str(n_stayed_less_than_025) + ', ' + str(n_stayed_less_than_025 + n_stayed_more_than_025)
            
            title += '\n' + 'Of the landings, ' + str(stayed_more_than_500_lower*100)[0:3] + '-' + str(stayed_more_than_500_upper*100)[0:3] + '\%' + ' stayed on the post for 5.0 sec or more'# + ' N: ' + str(n_stayed_less_than_025) + ', ' + str(n_stayed_less_than_025 + n_stayed_more_than_025)
            
            
            
            title += '\nMean trajectory length: ' + str(np.mean(trajec_lengths)) + ' +/- ' + str(np.std(trajec_lengths)) + ' sec'
            
            '''
            
            plot_all_heatmaps(config, dataset, save_figure_path=save_figure_path, figname=figname, title=title, keys=keys)
            pp.savefig()
            plt.close('all')
            
        pp.close()
    
    
    
if __name__ == '__main__':
    config = analysis_configuration.Config()
    culled_dataset = fad.load('../' + config.culled_datasets_path + config.culled_dataset_name)
    
    pdf_book(config, culled_dataset, save_figure_path='../../figures/heatmaps/')
    


