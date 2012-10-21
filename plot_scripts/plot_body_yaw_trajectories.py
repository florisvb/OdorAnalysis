#!/usr/bin/env python
import sys, os, imp
sys.path.append('../')
sys.path.append('../analysis_modules')
from optparse import OptionParser
import flydra_analysis_tools as fat
import fly_plot_lib
fly_plot_lib.set_params.pdf()
from matplotlib.backends.backend_pdf import PdfPages

import fly_plot_lib.plot as fpl
fad = fat.flydra_analysis_dataset
dac = fat.dataset_analysis_core
fap = fat.flydra_analysis_plot
tac = fat.trajectory_analysis_core

from flydra_analysis_tools import floris_math
from flydra_analysis_tools import kalman_math

import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

def get_keys_with_orientation_and_odor(dataset, keys=None):
    if keys is None:
        keys = dataset.trajecs.keys()
    keys_with_orientation_and_odor = []
    for key in keys:
        trajec = dataset.trajecs[key]
        
        try:
            if len(trajec.frames_with_orientation) > 30:
                if np.max(trajec.odor[trajec.frames_with_orientation]) > 50:
                    print key
                    keys_with_orientation_and_odor.append(key)
                    
        except:
            pass
    return keys_with_orientation_and_odor
        
        
        
#############################################################################################################################3
# Plot trajectories

def plot_trajectory_from_path(path):
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    
    culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
    dataset = fad.load(culled_dataset_filename)
    
    plot_trajectory(dataset, config)

def plot_trajectory(dataset, config, keys=None):
    path = config.path
    
    if keys is None:
        keys = get_keys_with_orientation_and_odor(dataset, keys=None)
    

    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'body_orientation_trajectories.pdf')
    pp = PdfPages(pdf_name_with_path)

    n_to_plot = 50
    n = -1
    for key in keys:
            
        trajec = dataset.trajecs[key]
        
        try:
            frames = trajec.frames_with_orientation
        except:
            continue
        
        if len(trajec.frames_with_orientation) < 5:
            continue
            
        n += 1
        if n >= n_to_plot:
            break
        print key
        
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_title(key.replace('_', '-'))
            
        ax.plot(trajec.positions[frames[0]-10:frames[-1]+10,0], -1*trajec.positions[frames[0]-10:frames[-1]+10,1], 'black', zorder=-100, linewidth=0.25)
        
        if 1:
            fpl.colorline_with_heading(ax,trajec.positions[frames,0], -1*trajec.positions[frames,1], trajec.odor[frames], orientation=trajec.orientation, colormap='jet', alpha=1, colornorm=[0,100], size_radius=0.15-np.abs(trajec.positions[frames,2]), size_radius_range=[.02, .02], deg=False, nskip=0, center_point_size=0.01, flip=False)
            
            ax.set_xlim(-.1, .3)
            ax.set_ylim(-.15, .15)
        
        if 0:
            x = []
            y = []
            for f in range(len(trajec.orientation_center)):
                x.append(trajec.orientation_center[f][1])
                y.append(trajec.orientation_center[f][0])
                
            fpl.colorline_with_heading(ax, np.array(x), np.array(y), trajec.odor[frames], orientation=trajec.orientation, colormap='jet', alpha=1, colornorm=[0,100], size_radius=0.15-np.abs(trajec.positions[frames,2]), size_radius_range=[10,10], deg=False, nskip=0, center_point_size=0.01, flip=False)
            
            ax.set_xlim(0,650)
            ax.set_ylim(0,480)
            
            
        pp.savefig()
        plt.close('all')

    pp.close()
