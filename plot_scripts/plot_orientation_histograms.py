import sys, os
from optparse import OptionParser
import pickle
import imp

from matplotlib.backends.backend_pdf import PdfPages

import fly_plot_lib
fly_plot_lib.set_params.pdf()
import fly_plot_lib.plot as fpl
import fly_plot_lib.animate as flyanim
import matplotlib.pyplot as plt

import flydra_analysis_tools.flydra_analysis_dataset as fad
from flydra_analysis_tools import floris_math
from flydra_analysis_tools import kalman_math

from flydra_analysis_tools import numpyimgproc as nim
import motmot.ufmf.ufmf as ufmf

import copy
import numpy as np


def get_orientation_data(dataset, config, visual_stimulus='none', odor_stimulus=None):

    threshold_odor = 10
    key_sets = {}
    
    if odor_stimulus is not None:
        keys_odor = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], [odor_stimulus, visual_stimulus])
    else:
        keys_odor = dataset.trajecs.keys()

    orientations = []
    eccentricities = []
    airheadings = []
    groundheadings = []
    speeds = []
    airspeeds = []
    
    keys = keys_odor
    
    for key in keys:
        trajec = dataset.trajecs[key]
        
        try:
            frames = trajec.frames_with_orientation
        except:
            continue
        
        if len(trajec.frames_with_orientation) < 5:
            continue
            
        frames_to_use = []
        saccade_frames = saccade_frames = [item for sublist in trajec.saccades for item in sublist]
        
        for i, f in enumerate(trajec.frames_with_orientation):
            if 1:
                if f in saccade_frames:
                    continue
                if np.abs(trajec.positions[f,2]) > 0.1:
                    continue
                if np.abs(trajec.velocities[f,2]) > 0.25:
                    continue
                if trajec.eccentricity[i] > 0.9:
                    continue
                
            frames_to_use.append(f)
            
        indices = [trajec.frames_with_orientation.index(f) for f in frames_to_use]
        
        orientations.extend(np.array(trajec.orientation)[indices].tolist())
        eccentricities.extend(np.array(trajec.eccentricity)[indices].tolist())
        airheadings.extend(trajec.airheading_smooth[frames_to_use].tolist())
        groundheadings.extend(trajec.heading_smooth[frames_to_use].tolist())
        speeds.extend(trajec.speed[frames_to_use].tolist())
        
        as_tmp = [np.linalg.norm(trajec.airvelocities[f]) for f in frames_to_use]
        airspeeds.extend(as_tmp)

    orientations = np.array(orientations)
    eccentricities = np.array(eccentricities)
    airheadings = np.array(airheadings)
    groundheadings = np.array(groundheadings)
    speeds = np.array(speeds)
    airspeeds = np.array(airspeeds)
    
    return orientations, airheadings, groundheadings, eccentricities, speeds, airspeeds
    

def plot_orientation_airheading_groundheading(dataset, config, visual_stimulus='none', odor_stimulus='on'):
    orientations, airheadings, groundheadings, eccentricities, speeds, airspeeds = get_orientation_data(dataset, config, visual_stimulus='none', odor_stimulus=odor_stimulus)
    
    data = {'orientation': orientations, 'airheadings': airheadings, 'groundheadings': groundheadings}
    #data = {'orientation': orientations-airheadings, 'airheadings': orientations-groundheadings, 'groundheadings': groundheadings}
    color = {'orientation': 'black', 'airheadings': 'green', 'groundheadings': 'red'}

    make_histograms(config, data, color)
    
def plot_slipangles(dataset, config, visual_stimulus='none', odor_stimulus='on'):
    orientations, airheadings, groundheadings, eccentricities, speeds, airspeeds = get_orientation_data(dataset, config, visual_stimulus='none', odor_stimulus=odor_stimulus)
    
    airslip = airheadings - orientations
    groundslip = groundheadings - orientations
    
    data = {'airslip': airslip, 'groundslip': groundslip}
    #data = {'orientation': orientations-airheadings, 'airheadings': orientations-groundheadings, 'groundheadings': groundheadings}
    color = {'airslip': 'green', 'groundslip': 'red'}
    
    savename = 'slipangle_histogram.pdf'

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)

    bins = np.linspace(-np.pi,np.pi,50)
        
    fpl.histogram(ax, data.values(), bins=bins, bin_width_ratio=1, colors=color.values(), edgecolor='none', bar_alpha=1, curve_fill_alpha=0.2, curve_line_alpha=1, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, smoothing_bins_to_exclude=[])
    
    xticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
    ax.set_xlim(xticks[0], xticks[-1])
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    
    ax.set_xlabel('slip angle')
    xticklabels = ['-180', '-90', '0', '90', '180']
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel('occurences, normalized')

    path = config.path
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    fig_name_with_path = os.path.join(save_figure_path, savename)

    print 'SAVING TO: ', fig_name_with_path
    fig.savefig(fig_name_with_path, format='pdf')


def plot_orientation_vs_groundheading(dataset, config, odor_stimulus='on'):
    orientations, airheadings, groundheadings, eccentricities, speeds, airspeeds = get_orientation_data(dataset, config, visual_stimulus='none', odor_stimulus=odor_stimulus)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_ylim(xticks[0], xticks[-1])
    
    fpl.scatter(ax, groundheadings, orientations, color=eccentricities, colornorm=[0,0.8], radius=.01, xlim=[xticks[0], xticks[-1]], ylim=[xticks[0], xticks[-1]])

    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=xticks)
    ax.set_xlabel('groundspeed heading')
    xticklabels = ['-180', '-90', 'upwind', '90', '180']
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(xticklabels)
    ax.set_ylabel('body orientation')
    
    savename = 'orientation_vs_groundheading.pdf'
    
    path = config.path
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    fig_name_with_path = os.path.join(save_figure_path, savename)

    print 'SAVING TO: ', fig_name_with_path
    fig.savefig(fig_name_with_path, format='pdf')
    

def plot_speed_vs_eccentricity(dataset, config):
    orientations, airheadings, groundheadings, eccentricities, speeds, airspeeds = get_orientation_data(dataset, config, visual_stimulus='none')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xticks = [0,1]
    yticks = [0,1]
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_ylim(yticks[0], yticks[-1])
    
    fpl.scatter(ax, speeds, eccentricities, color='black', colornorm=[0,0.8], radius=.003, xlim=[xticks[0], xticks[-1]], ylim=[yticks[0], yticks[-1]])

    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks)
    ax.set_xlabel('ground speed')
    ax.set_ylabel('eccentricity')
    
    savename = 'speed_vs_eccentricity.pdf'
    
    path = config.path
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    fig_name_with_path = os.path.join(save_figure_path, savename)

    print 'SAVING TO: ', fig_name_with_path
    fig.savefig(fig_name_with_path, format='pdf')
    

def make_histograms(config, data, colors, savename='orientation_histogram.pdf'):

    print data
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)

    bins = np.linspace(-np.pi,np.pi,50)
        
    fpl.histogram(ax, data.values(), bins=bins, bin_width_ratio=1, colors=colors.values(), edgecolor='none', bar_alpha=1, curve_fill_alpha=0.2, curve_line_alpha=1, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, smoothing_bins_to_exclude=[])
    
    xticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
    ax.set_xlim(xticks[0], xticks[-1])
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    
    ax.set_xlabel('heading')
    xticklabels = ['-180', '-90', 'upwind', '90', '180']
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel('occurences, normalized')

    path = config.path
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    fig_name_with_path = os.path.join(save_figure_path, savename)

    print 'SAVING TO: ', fig_name_with_path
    fig.savefig(fig_name_with_path, format='pdf')
    
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
    
    
    plot_orientation_airheading_groundheading(dataset, config, visual_stimulus='none')
    
            
