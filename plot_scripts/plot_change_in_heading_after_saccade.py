# heading during odor
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

import help_functions as hf


def plot_odor_heading_book(pp, threshold_odor, path, config, dataset, odor_stimulus, keys=None, axis='xy'):

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    

    saccade_angles_after_odor = []
    heading_at_saccade_initiation = []
    odor_at_saccade = []
    saccade_number = []
                
    if 1:
        for key in keys:
            trajec = dataset.trajecs[key]
            
            #if trajec.positions[0,0] < 0.2:
            #    continue
            
            frames_in_odor = np.where(trajec.odor > threshold_odor)[0]
            odor_blocks = hf.find_continuous_blocks(frames_in_odor, 5, return_longest_only=False)
            
            b = 0
            for block in odor_blocks:
                if len(block) < 5:
                    continue
                first_sac = None
                
                if axis == 'xy':
                    saccades = trajec.saccades
                elif axis == 'altitude':
                    saccades = trajec.saccades_z
                
                for sac in saccades:
                    if trajec.positions[sac[0],0] < -0.1 or trajec.positions[sac[0],0] > 0.9:
                        continue
                    if np.abs(trajec.positions[sac[0],1]) > 0.05:
                        continue
                    if trajec.positions[sac[0],2] > 0.05 or trajec.positions[sac[0],2] < -0.01:
                        continue
                        
                    if sac[0] > block[0]:
                        if first_sac is None:
                            if trajec.time_fly[sac[0]] - trajec.time_fly[block[-1]] > 0.5:
                                break
                            first_sac = sac
                            break
                        
                if first_sac is not None:
                    next_sac = first_sac
                    if axis == 'xy':
                        angle_of_saccade = tac.get_angle_of_saccade(trajec, next_sac)
                        heading_prior_to_saccade = trajec.heading_smooth[next_sac[0]]
                        # flip heading
                        if heading_prior_to_saccade < 0:
                            heading_prior_to_saccade += np.pi
                        else:
                            heading_prior_to_saccade -= np.pi
                    elif axis == 'altitude':
                        angle_of_saccade = tac.get_angle_of_saccade_z(trajec, next_sac)
                        heading_prior_to_saccade = trajec.heading_altitude_smooth[next_sac[0]]
                        # flip heading
                        if heading_prior_to_saccade < 0:
                            heading_prior_to_saccade += np.pi
                        else:
                            heading_prior_to_saccade -= np.pi
                    
                        
                    saccade_angles_after_odor.append(angle_of_saccade)
                    heading_at_saccade_initiation.append(heading_prior_to_saccade)
                    odor_at_saccade.append(trajec.odor[next_sac[0]])
                    b += 1
                    saccade_number.append(b)
        
    saccade_angles_after_odor = np.array(saccade_angles_after_odor)
    heading_at_saccade_initiation = np.array(heading_at_saccade_initiation)
    odor_at_saccade = np.array(odor_at_saccade)
    saccade_number = np.array(saccade_number)
    
    print odor_stimulus, saccade_angles_after_odor.shape
    
    #ax.plot(heading_at_saccade_initiation*180./np.pi, saccade_angles_after_odor*180./np.pi, '.', markersize=3)
    fpl.scatter(ax, heading_at_saccade_initiation*180./np.pi, saccade_angles_after_odor*180./np.pi, color=saccade_number, radius=3, colornorm=[0,5])
    #ax.plot(heading_at_saccade_initiation*180./np.pi, heading_after_saccade*180./np.pi, '.')
    
    xticks = [-180, -90, 0, 90, 180]
    yticks = [-180, -90, 0, 90, 180]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks)
    ax.set_xlabel('Heading before saccade')
    ax.set_ylabel('Angle of saccade')
    
    title_text = 'Odor: ' + odor_stimulus + ' Visual Stim: ' + trajec.visual_stimulus
    ax.set_title(title_text)
    
    ax.text(0,-180, 'Upwind', horizontalalignment='center', verticalalignment='top')
    ax.text(90,-180, 'Starboard', horizontalalignment='center', verticalalignment='top')
    ax.text(-90,-180, 'Port', horizontalalignment='center', verticalalignment='top')
    
    ax.text(-180,90, 'Starboard', horizontalalignment='left', verticalalignment='center', rotation='vertical')
    ax.text(-180,-90, 'Port', horizontalalignment='left', verticalalignment='center', rotation='vertical')
    
    pp.savefig()
    plt.close('all')
        

def pdf_book(config, dataset, save_figure_path='', axis='xy'):
    path = config.path
    if save_figure_path == '':
        figure_path = os.path.join(config.path, config.figure_path)
        save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'saccade_angle_after_odor_plume.pdf')
    pp = PdfPages(pdf_name_with_path)

    threshold_odor=30
    threshold_distance_min=.1
    visual_stimulus = ['none', 'upwind', 'downwind']
    
    for vstim in visual_stimulus:
    
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
                    
                    if trajec.visual_stimulus != vstim:
                        add_key = False
                    
                    if add_key:
                        keys.append(key)
                
                if len(keys) > 0:    
                    key_set.setdefault(odor_stimulus, keys)

            for odor_stimulus, keys in key_set.items():
                print 'Odor Book, Chapter: ', odor_stimulus
                
                #book_name = 'odor_headings_book_' + odor_stimulus + '_' + str(odor) + '.pdf'
                plot_odor_heading_book(pp, threshold_odor, path, config, dataset, odor_stimulus, keys=keys, axis=axis)
            
            
    pp.close()

def main(config, dataset, axis='xy'):
    pdf_book(config, dataset, save_figure_path='', axis=axis)

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

