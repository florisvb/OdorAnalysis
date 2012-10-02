# heading during odor
import os, sys
sys.path.append('../analysis_modules')
import fly_plot_lib
fly_plot_lib.set_params.pdf()
from matplotlib.backends.backend_pdf import PdfPages

import fly_plot_lib.plot as fpl
import odor_packet_analysis as opa

from flydra_analysis_tools import floris_math
import flydra_analysis_tools.trajectory_analysis_core as tac
import help_functions as hf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import xyz_axis_grid

import help_functions as hf
import copy


def plot_saccade_positions(pp, threshold_odor, path, config, dataset, odor_stimulus, keys=None):

    ax_xy, ax_xz, ax_yz = xyz_axis_grid.get_axes()
    

    if odor_stimulus is 'on':
        n = 0
        for key in keys:
            if n > 1000:
                break
            trajec = dataset.trajecs[key]
            frames_in_odor = np.where(trajec.odor > threshold_odor)[0]
            odor_blocks = hf.find_continuous_blocks(frames_in_odor, 5, return_longest_only=False)
            
            for block in odor_blocks:
                if len(block) < 5:
                    continue
                first_sac = None
                for sac in trajec.saccades:
                    if trajec.positions[sac[0],0] < -0.1 or trajec.positions[sac[0],0] > 0.9:
                        continue
                    if np.abs(trajec.positions[sac[0],1]) > 0.05:
                        continue
                    if trajec.positions[sac[0],2] > 0.05 or trajec.positions[sac[0],2] < -0.01:
                        continue
                        
                    if sac[0] > block[0]:
                        if first_sac is None:
                            first_sac = sac
                            break
                        
                if first_sac is not None:
                    sac = first_sac
                    n += 1
                    f = np.argmax(np.abs(trajec.heading_smooth_diff[sac])) + sac[0]
                    
                    ax_xy.plot(trajec.positions[f,0], trajec.positions[f,1], '.', color='black')
                    ax_xz.plot(trajec.positions[f,0], trajec.positions[f,2], '.', color='black')
                    ax_yz.plot(trajec.positions[f,1], trajec.positions[f,2], '.', color='black')
    
    if odor_stimulus is 'none':
        n = 0
        for key in keys:
            if n > 1000:
                break
            trajec = dataset.trajecs[key]
            
            for sac in trajec.saccades:
                if trajec.positions[sac[0],0] < -0.1 or trajec.positions[sac[0],0] > 0.9:
                    continue
                if np.abs(trajec.positions[sac[0],1]) > 0.05:
                    continue
                if trajec.positions[sac[0],2] > 0.05 or trajec.positions[sac[0],2] < -0.01:
                    continue
                        
                n += 1
                f = np.argmax(np.abs(trajec.heading_smooth_diff[sac])) + sac[0]
                
                ax_xy.plot(trajec.positions[f,0], trajec.positions[f,1], '.', color='black')
                ax_xz.plot(trajec.positions[f,0], trajec.positions[f,2], '.', color='black')
                ax_yz.plot(trajec.positions[f,1], trajec.positions[f,2], '.', color='black')
    
    xyz_axis_grid.set_spines_and_labels(ax_xy, ax_xz, ax_yz)
    pp.savefig()
    plt.close('all')
        



def pdf_book(config, dataset, save_figure_path=''):
    path = config.path
    if save_figure_path == '':
        figure_path = os.path.join(config.path, config.figure_path)
        save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'saccade_positions.pdf')
    pp = PdfPages(pdf_name_with_path)

    threshold_odor=50
    
    for odor in [True]:
        
        key_set = {}
        for odor_stimulus in config.odor_stimulus.keys():
            #keys_tmp = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=threshold_odor, odor_stimulus=odor_stimulus, threshold_distance_min=threshold_distance_min, odor=odor)
            
            
            keys = []
            for key in dataset.trajecs.keys():
                trajec = dataset.trajecs[key]
                add_key = True
                
                if odor:
                    if np.max(trajec.odor) > threshold_odor:
                        frames_in_odor = np.where(trajec.odor > threshold_odor)[0]
                        if len(frames_in_odor) < 5:
                            add_key = False
                    else:
                        add_key = False
                if trajec.odor_stimulus != odor_stimulus:
                    add_key = False
                
                if add_key:
                    keys.append(key)
            
            if len(keys) > 0:    
                key_set.setdefault(odor_stimulus, keys)

        for odor_stimulus, keys in key_set.items():
            print 'Odor Book, Chapter: ', odor_stimulus
            
            #book_name = 'odor_headings_book_' + odor_stimulus + '_' + str(odor) + '.pdf'
            plot_saccade_positions(pp, threshold_odor, path, config, dataset, odor_stimulus, keys=keys)
            
            
    pp.close()

def main(config, dataset):
    pdf_book(config, dataset, save_figure_path='')

if __name__ == '__main__':
    pdf_book(config, dataset, save_figure_path='')

