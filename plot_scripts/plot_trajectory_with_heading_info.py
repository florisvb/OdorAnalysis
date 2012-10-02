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


def plot_trajectory(trajec):
    threshold_odor = 10
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    
    frames_in_odor = np.where(trajec.odor > threshold_odor)[0]
    odor_blocks = hf.find_continuous_blocks(frames_in_odor, 5, return_longest_only=False)
    
    for block in [odor_blocks[3]]:
        #middle_of_block = int(np.mean(block))
        if len(block) < 5:
            continue
        # find next saccade
        first_sac = None
        #second_sac = None
        #third_sac = None
        for sac in trajec.saccades:
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
            
            if heading_prior_to_saccade < 0:
                heading_prior_to_saccade += np.pi
            else:
                heading_prior_to_saccade -= np.pi
            
        frame0 = np.max([next_sac[0]-20, 0])
        frame1 = np.min([next_sac[0]+20, trajec.length-1])
        frames = np.arange(frame0, frame1)
        
        ax.plot(trajec.positions[frames,0], trajec.positions[frames,1])
        ax.plot(trajec.positions[frames[0],0], trajec.positions[frames[0],1], '.', color='green')
        #pos_before_sac = trajec.positions[next_sac[0], :]
        #heading_vector = pos_before_sac
        print 'raw heading prior: ', heading_prior_to_saccade*180/np.pi
        print 'raw heading after: ', trajec.heading_smooth[next_sac[-1]]*180/np.pi
        
        print 'raw angle of sac: ', angle_of_saccade*180/np.pi
    
    ax.set_aspect('equal')
    return next_sac
    
    
if __name__ == '__main__':
    
    next_sac = plot_trajectory(trajec)
