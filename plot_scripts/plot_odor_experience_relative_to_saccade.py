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


def plot_odor_heading_book(pp, threshold_odor, path, config, dataset, odor_stimulus, keys=None):

    fig = plt.figure(figsize=(8,5))
    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    ax_odor = fig.add_subplot(511)
    ax_wall = fig.add_subplot(512)
    ax_z = fig.add_subplot(513)
    ax_heading = fig.add_subplot(514)
    ax_speed = fig.add_subplot(515)
    
    axes = [ax_odor, ax_wall, ax_z, ax_heading, ax_speed]
    
    if odor_stimulus is 'pulsing':
        return

    if 1: #odor_stimulus is 'on':
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
                        
                    if sac[0] > block[-1] and trajec.odor[sac[0]] < threshold_odor:
                        if first_sac is None:
                            first_sac = sac
                            break
                        
                if first_sac is not None:
                    sac = first_sac
                    n += 1
                    
                    heading_prior_to_saccade = trajec.heading_smooth[sac[0]]
                    # flip heading
                    if heading_prior_to_saccade < 0:
                        heading_prior_to_saccade += np.pi
                    else:
                        heading_prior_to_saccade -= np.pi
                        
                    #if np.abs(heading_prior_to_saccade) > 45*np.pi/180. and np.abs(heading_prior_to_saccade) < 135*np.pi/180.:
                    if np.abs(heading_prior_to_saccade) < 15*np.pi/180.:# and np.abs(heading_prior_to_saccade) < 135*np.pi/180.:
                        
                        if 1:
                            frame_of_saccade_middle = np.argmax(np.abs(trajec.heading_smooth_diff[sac])) + sac[0]
                            time_of_sac = trajec.time_fly[frame_of_saccade_middle]
                        
                        if 0:
                            frame_of_max_odor = np.argmax(trajec.odor[block])+block[0]
                            time_of_sac = trajec.time_fly[frame_of_max_odor]

                        frame0 = np.max([0,sac[0]-60])
                        frame1 = np.min([trajec.length-1,sac[-1]+100])
                        
                        if frame0 > 0:
                            frames = np.arange(frame0, frame1)
                            
                            # speed
                            # in saccade:
                            ax_speed.plot(trajec.time_fly[sac]-time_of_sac, trajec.speed[sac], 'crimson', linewidth=0.5, alpha=0.25)
                            # before saccade
                            frames_before_sac = np.arange(frames[0], sac[1])
                            ax_speed.plot(trajec.time_fly[frames_before_sac]-time_of_sac, trajec.speed[frames_before_sac], 'black', linewidth=0.5, alpha=0.25)
                            # after saccade
                            frames_after_sac = np.arange(sac[-2],frames[-1])
                            ax_speed.plot(trajec.time_fly[frames_after_sac]-time_of_sac, trajec.speed[frames_after_sac], 'black', linewidth=0.5, alpha=0.25)
                            
                            # wall
                            # in saccade:
                            ax_wall.plot(trajec.time_fly[sac]-time_of_sac, np.abs(trajec.positions[sac,1]), 'crimson', linewidth=0.5, alpha=0.25)
                            # before saccade
                            ax_wall.plot(trajec.time_fly[frames_before_sac]-time_of_sac, np.abs(trajec.positions[frames_before_sac,1]), 'black', linewidth=0.5, alpha=0.25)
                            # after saccade
                            ax_wall.plot(trajec.time_fly[frames_after_sac]-time_of_sac, np.abs(trajec.positions[frames_after_sac,1]), 'black', linewidth=0.5, alpha=0.25)
                            
                            # z
                            # in saccade:
                            ax_z.plot(trajec.time_fly[sac]-time_of_sac, (trajec.positions[sac,2]), 'crimson', linewidth=0.5, alpha=0.25)
                            # before saccade
                            ax_z.plot(trajec.time_fly[frames_before_sac]-time_of_sac, (trajec.positions[frames_before_sac,2]), 'black', linewidth=0.5, alpha=0.25)
                            # after saccade
                            ax_z.plot(trajec.time_fly[frames_after_sac]-time_of_sac, (trajec.positions[frames_after_sac,2]), 'black', linewidth=0.5, alpha=0.25)
                            
                            
                            # heading
                            heading = copy.copy(trajec.heading_smooth)
                            for i, h in enumerate(heading):
                                if h < 0:
                                    heading[i] += np.pi
                                else:
                                    heading[i] -= np.pi
                            #heading = floris_math.remove_angular_rollover(heading, np.pi)
                            # in saccade:
                            ax_heading.plot(trajec.time_fly[sac]-time_of_sac, np.abs(heading[sac]), 'crimson', linewidth=0.5, alpha=0.25)
                            # before saccade
                            ax_heading.plot(trajec.time_fly[frames_before_sac]-time_of_sac, np.abs(heading[frames_before_sac]), 'black', linewidth=0.5, alpha=0.25)
                            # after saccade
                            ax_heading.plot(trajec.time_fly[frames_after_sac]-time_of_sac, np.abs(heading[frames_after_sac]), 'black', linewidth=0.5, alpha=0.25)
                            
                            # odor
                            # in saccade:
                            ax_odor.plot(trajec.time_fly[sac]-time_of_sac, trajec.odor[sac], 'crimson', linewidth=0.5, alpha=0.25)
                            # before saccade
                            ax_odor.plot(trajec.time_fly[frames_before_sac]-time_of_sac, trajec.odor[frames_before_sac], 'black', linewidth=0.5, alpha=0.25)
                            # after saccade
                            ax_odor.plot(trajec.time_fly[frames_after_sac]-time_of_sac, trajec.odor[frames_after_sac], 'black', linewidth=0.5, alpha=0.25)
                            
                            
                            ######################################3
    
    for ax in axes:
        ax.set_xlim(-1,2)
    
    ax_speed.set_ylim(0,.8)
    yticks = [0,.4,.8]
    fpl.adjust_spines(ax_speed, ['left', 'bottom'], yticks=yticks)
    ax_speed.set_xlabel('time relative to saccade, sec')
    ax_speed.set_ylabel('speed, m/s')
    
    yticks = [0,125,250]
    fpl.adjust_spines(ax_odor, ['left'], yticks=yticks)
    ax_odor.set_ylabel('est. odor', horizontalalignment='center')
    
    yticks = [-.15, 0, .15]
    ax_wall.set_ylim(-.15, 0, .15)
    fpl.adjust_spines(ax_wall, ['left'], yticks=yticks)
    ax_wall.set_ylabel('y pos, m', horizontalalignment='center')
    
    yticks = [-.15, 0, .15]
    ax_z.set_ylim(-.15,0.15)
    fpl.adjust_spines(ax_z, ['left'], yticks=yticks)
    ax_z.set_ylabel('altitude, m')
    
    ax_heading.set_ylim(0, np.pi+np.pi/6.)
    yticks = [0, np.pi/2., np.pi]
    fpl.adjust_spines(ax_heading, ['left'], yticks=yticks)
    ax_heading.set_yticklabels(['upwind', 'crosswind', 'downwind'])
    ax_heading.set_ylabel('heading')
    
    title_str = 'Odor Stimulus: ' + odor_stimulus
    fig.suptitle(title_str)
    
    
    
    pp.savefig()
    plt.close('all')
        



def pdf_book(config, dataset, save_figure_path=''):
    path = config.path
    if save_figure_path == '':
        figure_path = os.path.join(config.path, config.figure_path)
        save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'odor_experience_before_after_saccade.pdf')
    pp = PdfPages(pdf_name_with_path)

    threshold_odor=10
    
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
            plot_odor_heading_book(pp, threshold_odor, path, config, dataset, odor_stimulus, keys=keys)
            
            
    pp.close()

def main(config, dataset):
    pdf_book(config, dataset, save_figure_path='')

if __name__ == '__main__':
    pdf_book(config, dataset, save_figure_path='')

