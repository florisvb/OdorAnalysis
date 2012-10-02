# heading during odor
import os, sys
sys.path.append('../analysis_modules')
import fly_plot_lib
fly_plot_lib.set_params.pdf()
from matplotlib.backends.backend_pdf import PdfPages

import fly_plot_lib.plot as fpl
import odor_packet_analysis as opa

import flydra_analysis_tools.trajectory_analysis_core as tac
import help_functions as hf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import xyz_axis_grid

import help_functions as hf


def plot_odor_heading_book(pp, threshold_odor, path, config, dataset, odor_stimulus, keys=None):

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    

    saccade_angles_after_odor = []
    heading_at_saccade_initiation = []
    heading_after_saccade = []
    speed_at_saccade = []
    for key in keys:
        trajec = dataset.trajecs[key]
        frames_in_odor = np.where(trajec.odor > threshold_odor)[0]
        odor_blocks = hf.find_continuous_blocks(frames_in_odor, 5, return_longest_only=False)
        
        for block in odor_blocks:
            #middle_of_block = int(np.mean(block))
            if len(block) < 5:
                continue
            # find next saccade
            first_sac = None
            #second_sac = None
            #third_sac = None
            for sac in trajec.saccades:
                if trajec.positions[sac[0],0] < -0.15 or trajec.positions[sac[0],0] > 0.9:
                    continue
                if np.abs(trajec.positions[sac[0],1]) > 0.08:
                    continue
                if np.abs(trajec.positions[sac[0],2]) > 0.08:
                    continue
                    
                
                if sac[0] > block[0]:
                    if first_sac is None:
                        first_sac = sac
                        break
                    #elif second_sac is None:
                    #    if trajec.odor[sac[0]] < threshold_odor:
                    #        second_sac = sac
                    #elif third_sac is None:
                    #    if trajec.odor[sac[0]] < threshold_odor:
                    #        third_sac = sac
                    #    break
                
                    
            if first_sac is not None:
                next_sac = first_sac
                angle_of_saccade = tac.get_angle_of_saccade(trajec, next_sac)
                heading_prior_to_saccade = trajec.heading_smooth[next_sac[0]]
                # flip heading
                if heading_prior_to_saccade < 0:
                    heading_prior_to_saccade += np.pi
                else:
                    heading_prior_to_saccade -= np.pi
                
                    
                saccade_angles_after_odor.append(angle_of_saccade)
                heading_at_saccade_initiation.append(heading_prior_to_saccade)
                heading_after_saccade.append(heading_prior_to_saccade + angle_of_saccade)
                speed_at_saccade.append(trajec.speed[next_sac[0]])
        
    saccade_angles_after_odor = np.array(saccade_angles_after_odor)
    heading_at_saccade_initiation = np.array(heading_at_saccade_initiation)
    heading_after_saccade = np.array(heading_after_saccade)
    speed_at_saccade = np.array(speed_at_saccade)
    
    ax.plot(heading_at_saccade_initiation*180./np.pi, saccade_angles_after_odor*180./np.pi, '.', markersize=4)
    #fpl.scatter(ax, heading_at_saccade_initiation*180./np.pi, saccade_angles_after_odor*180./np.pi, color=speed_at_saccade, radius=1)
    #ax.plot(heading_at_saccade_initiation*180./np.pi, heading_after_saccade*180./np.pi, '.')
    
    xticks = [-180, -90, 0, 90, 180]
    yticks = [-180, -90, 0, 90, 180]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks)
    ax.set_xlabel('Heading before saccade')
    ax.set_ylabel('Angle of saccade')
    
    title_text = 'Odor: ' + trajec.odor_stimulus.title()
    ax.set_title(title_text)
    
    ax.text(0,-180, 'Upwind', horizontalalignment='center', verticalalignment='top')
    ax.text(90,-180, 'Starboard', horizontalalignment='center', verticalalignment='top')
    ax.text(-90,-180, 'Port', horizontalalignment='center', verticalalignment='top')
    
    ax.text(-180,90, 'Starboard', horizontalalignment='left', verticalalignment='center', rotation='vertical')
    ax.text(-180,-90, 'Port', horizontalalignment='left', verticalalignment='center', rotation='vertical')
    
    pp.savefig()
    plt.close('all')
        

    # angle of saccade histogram
    if 0:
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        
        fpl.histogram_stack(ax, [np.array(time_to_saccade_cast)], bins=40, bin_width_ratio=0.9, colors=['red'], edgecolor='none', normed=True)

        ax.set_xlabel('Angle of Saccade')
        ax.set_ylabel('Occurences, normalized')
        #xticks = [-180, -90, 0, 90, 180]
        fpl.adjust_spines(ax, ['left', 'bottom'])

        ax.set_title(title_text)

        pp.savefig()
        plt.close('all')




def pdf_book(config, dataset, save_figure_path=''):
    path = config.path
    if save_figure_path == '':
        figure_path = os.path.join(config.path, config.figure_path)
        save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'saccade_angle_after_odor_plume.pdf')
    pp = PdfPages(pdf_name_with_path)

    threshold_odor=10
    threshold_distance_min=.1
    
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
                        if len(frames_in_odor) < 40:
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
            plot_odor_heading_book(pp, threshold_odor, path, config, dataset, keys=keys)
            
            
    pp.close()

def main(config, dataset):
    pdf_book(config, dataset, save_figure_path='')

if __name__ == '__main__':
    pdf_book(config, dataset, save_figure_path='')

