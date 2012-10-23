import os, sys
sys.path.append('../analysis_modules')
import imp
from optparse import OptionParser
import fly_plot_lib
fly_plot_lib.set_params.pdf()
from matplotlib.backends.backend_pdf import PdfPages

import fly_plot_lib.plot as fpl
import fly_plot_lib.flymath as flymath
import odor_packet_analysis as opa

import flydra_analysis_tools.trajectory_analysis_core as tac
import flydra_analysis_tools.flydra_analysis_dataset as fad

import numpy as np
import fly_plot_lib.plot as fpl

import matplotlib.pyplot as plt
import os

import matplotlib.patches as patches

import flydra_analysis_tools.trajectory_analysis_core as tac

import copy
from flydra_analysis_tools import floris_math
from flydra_analysis_tools import kalman_math



def get_odor_encounter_frames(trajec, threshold_odor=10):
    frames_where_odor = np.where(trajec.odor>threshold_odor)[0]
        
    entering_odor = []
    leaving_odor = []
    in_odor = False
    
    for f in range(trajec.length):
    
        if in_odor is False:
            if trajec.odor[f] > threshold_odor:
                entering_odor.append(f)
                in_odor = True
    
        elif in_odor:
            if trajec.odor[f] < threshold_odor:
                leaving_odor.append(f)
                in_odor = False
                
    return entering_odor, leaving_odor

def get_frame_chunks_relative_to_odor(trajec, threshold_odor=10, framerange=None):
    entering_odor, leaving_odor = get_odor_encounter_frames(trajec, threshold_odor=threshold_odor)
    
    frame_chunks_prior_odor = {}
    frame_chunks_after_odor = {}
    
    if np.max(trajec.odor) < threshold_odor:
        return frame_chunks_prior_odor, frame_chunks_after_odor
    
    frames_prior = []
    frames_after = []
    
    frame_entered = 0
    frame_left = 0
    
    if framerange is None:
        framerange = [0,trajec.length]
    frame_left_timer = None
    
    for frame in range(framerange[0], framerange[-1]):
        
        if frame in entering_odor: # start collecting frames for chunks after odor
            #print 'entered'
            #print len(frames_prior)
            #    print frames_prior
            #    frame_chunks_prior_odor.setdefault(frame_entered, copy.copy(frames_prior))
            #    frames_prior = []
                
            frame_entered = frame
            frame_chunks_after_odor.setdefault(frame_left, copy.copy(frames_after))
            frames_after = []
            frame_left_timer = None
        
        if frame in leaving_odor:
            frame_left = frame
            frame_left_timer = 0
            
        if frame_left_timer is not None:
            frame_left_timer += 1
        if frame_left_timer >= 50:
            frame_chunks_prior_odor.setdefault(frame_entered, copy.copy(frames_prior))
            frames_prior = []
            frame_left_timer = None
        
        frames_prior.append(frame)
        frames_after.append(frame)
    
    return frame_chunks_prior_odor, frame_chunks_after_odor
    
    
def get_attribute_and_time_relative_to_odor(trajec, threshold_odor=10, attribute='heading_smooth', column=0, relative_to='entry', velocity_adjustment=0, auto_velocity_adjustment=False, framerange=None):
    if np.max(trajec.odor) < threshold_odor:
        return [], []
        
    frame_chunks_prior_odor, frame_chunks_after_odor = get_frame_chunks_relative_to_odor(trajec, threshold_odor=threshold_odor, framerange=framerange)
    
    attribute_vals = []
    relative_time = []
    
    if relative_to == 'entry':
        chunks = frame_chunks_prior_odor
    elif relative_to == 'exit':
        chunks = frame_chunks_after_odor
        
    n = 0
    for frame_rel, frame_chunk in chunks.items():
        if n == 0:
            pass
            
        if 0:
            if relative_to=='entry':
                if frame_rel < 30:
                    continue
                if np.mean(np.abs(trajec.heading_smooth[frame_rel-30:frame_rel])) < 40*np.pi/180.:
                    continue
                
        if n >= 1:        
            for f in frame_chunk:
                #headings.append(trajec.heading_smooth[f])

                if attribute == 'velocities':
                    val = trajec.velocities[f,column]
                    val = np.arctan2( val, trajec.speed_xy[f] )
                elif attribute == 'heading_smooth':
                
                    if auto_velocity_adjustment:
                        if trajec.visual_stimulus == 'upwind':
                            velocity_adjustment = 0.08
                        elif trajec.visual_stimulus == 'downwind':
                            velocity_adjustment = -0.08
                        else:
                            velocity_adjustment = 0
                
                    if velocity_adjustment != 0:
                        velocities = copy.copy(trajec.velocities[:,0:2])
                        velocities[:,0] += velocity_adjustment
                        heading_smooth = calc_heading(velocities)
                    else:
                        heading_smooth = trajec.heading_smooth
                    val = heading_smooth[f]
                elif attribute == 'speed':
                    val = trajec.speed[f]
                    
                elif attribute == 'heading_smooth_diff':
                    val = trajec.heading_smooth_diff[f]
                    
                else:
                    print 'edit the code.. '
                        
                attribute_vals.append(val)
                relative_time.append(trajec.time_fly[f] - trajec.time_fly[frame_rel])
        n += 1
    
    return attribute_vals, relative_time
    

def get_attribute_and_time_relative_to_odor_for_dataset(dataset, keys=None, threshold_odor=10, attribute='heading_smooth', column=0, relative_to='entry', velocity_adjustment=0, auto_velocity_adjustment=False, framerange=None):
    
    attribute_vals = []
    relative_time = []
    
    if keys is None:
        keys = dataset.trajecs.keys()
        
    for key in keys:
        print key
        trajec = dataset.trajecs[key]
        a, r = get_attribute_and_time_relative_to_odor(trajec, threshold_odor=threshold_odor, attribute=attribute, column=column, relative_to=relative_to, velocity_adjustment=velocity_adjustment, auto_velocity_adjustment=auto_velocity_adjustment, framerange=framerange)
        attribute_vals.extend(a)
        relative_time.extend(r)
        
    return attribute_vals, relative_time
    

def plot_heading_time_to_odor_on_ax(ax, dataset, keys=None, threshold_odor=10, attribute='heading_smooth', column=0, xlim=[-np.pi, np.pi], relative_to='entry', velocity_adjustment=0, auto_velocity_adjustment=False, framerange=None):
    
    #speed_vals, relative_time = get_attribute_and_time_relative_to_odor_for_dataset(dataset, keys=keys, threshold_odor=threshold_odor, attribute='speed', column=column, relative_to=relative_to)
    #speed_vals = np.array(speed_vals)
    
    attribute_vals, relative_time = get_attribute_and_time_relative_to_odor_for_dataset(dataset, keys=keys, threshold_odor=threshold_odor, attribute=attribute, column=column, relative_to=relative_to, velocity_adjustment=velocity_adjustment, auto_velocity_adjustment=auto_velocity_adjustment, framerange=framerange)
    attribute_vals = np.array(attribute_vals)
    relative_time = np.array(relative_time)
    
    #ax.set_rasterization_zorder(100)
    
    # handle time before odor
    indices = np.where(relative_time<0)[0]
    attribute_vals_prior = attribute_vals[indices]
    time_prior = relative_time[indices]
    time_prior = -1*np.log(-1*time_prior + 1)
    binsx = np.linspace(xlim[0], xlim[1],100)
    binsy = np.linspace(-1*np.log(10),0,100)
    #fpl.scatter(ax, attribute_vals_prior, time_prior, color=speed_vals, radius=.001, colornorm=[0,.6])
    fpl.histogram2d(ax, attribute_vals_prior, time_prior, bins=(binsx, binsy), logcolorscale=True)
    
    # handle time after odor
    indices = np.where(relative_time>0)[0]
    attribute_vals_after = attribute_vals[indices]
    time_after = relative_time[indices]
    time_after = np.log(time_after + 1)
    binsx = np.linspace(xlim[0], xlim[1],100)
    binsy = np.linspace(0,np.log(9),100)
    #fpl.scatter(ax, attribute_vals_after, time_after, color=speed_vals, radius=.001, colornorm=[0,.6])
    fpl.histogram2d(ax, attribute_vals_after, time_after, bins=(binsx, binsy), logcolorscale=True)
    
def plot_heading_time_to_odor_for_trajec_on_ax(ax, trajec, threshold_odor=10, attribute='heading_smooth', column=0, relative_to='entry', framerange=None):
        attribute, relative_time = get_attribute_and_time_relative_to_odor(trajec, threshold_odor=threshold_odor, attribute=attribute, column=column, relative_to=relative_to, framerange=framerange)
        if len(relative_time) < 10:
            return
        time_chunks, attribute_chunks, breakpoints = flymath.get_continuous_chunks(relative_time, attribute, jump=0.02)
        print trajec.key
        
        for i, relative_time in enumerate(time_chunks):
            print i
            attribute_vals = np.array(attribute_chunks[i])
            relative_time = np.array(relative_time)
            
            # handle time before odor
            indices = np.where(relative_time<0)[0]
            attribute_vals_prior = attribute_vals[indices]
            time_prior = relative_time[indices]
            time_prior = -1*np.log(-1*time_prior + 1)
            # handle time after odor
            indices = np.where(relative_time>0)[0]
            attribute_vals_after = attribute_vals[indices]
            time_after = relative_time[indices]
            time_after = np.log(time_after + 1)
            
            # merge them
            time_merged = np.hstack((time_prior, time_after))
            attribute_vals_merged = np.hstack((attribute_vals_prior, attribute_vals_after))
            
            time_merged, attribute_vals_merged = flymath.remove_discontinuities(time_merged, attribute_vals_merged, jump=2)
            ax.plot(attribute_vals_merged, time_merged)

def plot_heading_time_to_odor_relative_for_trajec(config, dataset, keys, attribute='heading_smooth', column=0, relative_to='entry', threshold_odor=10, framerange=None):

    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    if attribute == 'heading_smooth':
        xlim = [-np.pi, np.pi]
        ylim = [-1*np.log(10), np.log(10)]
        xticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
        xticklabels = [-180,-90,0,90,180]
        xlabel = 'heading'
    elif attribute == 'velocities':
        xlim = [-1.5, 1.5]
        ylim = [-1*np.log(10), np.log(10)]
        xticks = [-1.5, 0, 1.5]
        xticklabels = [-1.5, 0, 1.5]
        xlabel = 'vertical velocity'
    elif attribute == 'speed':
        xlim = [-1, 1]
        ylim = [-1*np.log(10), np.log(10)]
        xticks = [-1, 0, 1]
        xticklabels = [-1, 0, 1]
        xlabel = 'speed'
    elif attribute == 'heading_smooth_diff':
        print attribute
        xlim = [-1500*np.pi/180, 1500*np.pi/180]
        ylim = [-1*np.log(10), np.log(10)]
        xticks = [-1500*np.pi/180, 0, 1500*np.pi/180]
        xticklabels = [-1500, 0, 1500]
        xlabel = 'angular velocity'
    else:
        print 'Choose a valid attribute!'
        
    ax = fig.add_subplot(111)
    
    if 0:
        yticks_neg = [-9,-8,-7,-6,-5,-4,-3,-2,-1,0]
        yticks_neg = -1*np.log(-1*np.array(yticks_neg)+1)
        yticks_pos = [1,2,3,4,5,6,7,8,9]
        yticks_pos = np.log(np.array(yticks_pos)+1)
        yticks = yticks_neg.tolist()
        yticks.extend(yticks_pos.tolist())
        ytick_labels = [-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]
    if 1:
        yticks = [-np.log(2),0,np.log(2)]
        ytick_labels = [-2,0,2]
        ylim = [-np.log(2),np.log(2)]
    
    
    for key in keys:
        trajec = dataset.trajecs[key]
        plot_heading_time_to_odor_for_trajec_on_ax(ax, trajec, threshold_odor=threshold_odor, attribute=attribute, column=column, relative_to=relative_to, framerange=framerange)
    
    ax.set_aspect('auto')
    #ax.set_ylim(ylim[0], ylim[1])
    
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel(xlabel)
    
    if relative_to == 'entry':
        ax.set_ylabel('time relative to odor entry, sec')
    else:
        ax.set_ylabel('time relative to odor exit, sec')
    
    ax.set_aspect('auto')
    ax.set_title(str(trajec.key).replace('_','-'))
    
    path = config.path
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    figname = attribute + '_vs_time_' + 'relative_to_' + relative_to + '_odor_for_trajec.pdf'
    pdf_name_with_path = os.path.join(save_figure_path, figname)
    
    print 'SAVING TO: ', pdf_name_with_path
    
    fig.savefig(pdf_name_with_path, format='pdf')
    
    
        
def plot_heading_time_to_odor_relative(config, dataset, attribute='heading_smooth', column=0, relative_to='entry', threshold_odor=10):
    
    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    if attribute == 'heading_smooth':
        xlim = [-np.pi, np.pi]
        ylim = [-1*np.log(10), np.log(10)]
        xticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
        xticklabels = [-180,-90,0,90,180]
        xlabel = 'heading'
    if attribute == 'velocities':
        xlim = [-1.5, 1.5]
        ylim = [-1*np.log(10), np.log(10)]
        xticks = [-1.5, 0, 1.5]
        xticklabels = [-1.5, 0, 1.5]
        xlabel = 'vertical velocity'
    if attribute == 'speed':
        xlim = [-1, 1]
        ylim = [-1*np.log(10), np.log(10)]
        xticks = [-1, 0, 1]
        xticklabels = [-1, 0, 1]
        xlabel = 'speed'
    if attribute == 'heading_smooth_diff':
        print attribute
        xlim = [-1500*np.pi/180, 1500*np.pi/180]
        ylim = [-1*np.log(10), np.log(10)]
        xticks = [-1500*np.pi/180, 0, 1500*np.pi/180]
        xticklabels = [-1500, 0, 1500]
        xlabel = 'angular velocity'
    else:
        print 'Choose a valid attribute!'
        
    keys_noodor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'none')
    ax = fig.add_subplot(121)
    yticks_neg = [-9,-8,-7,-6,-5,-4,-3,-2,-1,0]
    yticks_neg = -1*np.log(-1*np.array(yticks_neg)+1)
    yticks_pos = [1,2,3,4,5,6,7,8,9]
    yticks_pos = np.log(np.array(yticks_pos)+1)
    yticks = yticks_neg.tolist()
    yticks.extend(yticks_pos.tolist())
    ytick_labels = [-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]
    
    ax.set_aspect('auto')
    ax.set_ylim(ylim[0], ylim[1])
    
    plot_heading_time_to_odor_on_ax(ax, dataset, keys=keys_noodor, attribute=attribute, column=column, xlim=xlim, relative_to=relative_to, threshold_odor=threshold_odor)
    ax.vlines(-np.pi/2., -10, 5, color='black')
    ax.vlines(np.pi/2., -10, 5, color='black')
    ax.set_title('odor off')
    
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel(xlabel)
    
    if relative_to == 'entry':
        ax.set_ylabel('time relative to odor entry, sec')
    else:
        ax.set_ylabel('time relative to odor exit, sec')
    
    ax.set_aspect('auto')

    ##########
    keys_odor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'on')
    ax = fig.add_subplot(122)
    keys_to_show = ['20120926_174720_6196']
    
    ax.set_aspect('auto')
    ax.set_ylim(ylim[0], ylim[1])
    
    plot_heading_time_to_odor_on_ax(ax, dataset, keys=keys_odor, attribute=attribute, column=column, xlim=xlim, relative_to=relative_to, threshold_odor=threshold_odor)
    ax.vlines(-np.pi/2., -10, 5, color='black')
    ax.vlines(np.pi/2., -10, 5, color='black')
    ax.set_title('odor on')
    
    fpl.adjust_spines(ax, ['bottom'], xticks=xticks, yticks=yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_aspect('auto')
    
    path = config.path
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    figname = attribute + '_vs_time_' + 'relative_to_' + relative_to + '_odor' + '_threshold_' + str(threshold_odor) + '.pdf'
    pdf_name_with_path = os.path.join(save_figure_path, figname)
    
    print 'SAVING TO: ', pdf_name_with_path
    
    fig.savefig(pdf_name_with_path, format='pdf')


###################################################################################################################################
# Moving Floor Stuff
###################################################################################################################################

def plot_heading_time_to_odor_relative_moving_floor(config, dataset, attribute='heading_smooth', column=0, relative_to='entry', threshold_odor=10):
    
    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    if attribute == 'heading_smooth':
        xlim = [-np.pi, np.pi]
        ylim = [-1*np.log(10), np.log(10)]
        xticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
        xticklabels = [-180,-90,0,90,180]
        xlabel = 'heading'
    if attribute == 'velocities':
        xlim = [-1.5, 1.5]
        ylim = [-1*np.log(10), np.log(10)]
        xticks = [-1.5, 0, 1.5]
        xticklabels = [-1.5, 0, 1.5]
        xlabel = 'vertical velocity'
    if attribute == 'speed':
        xlim = [-1, 1]
        ylim = [-1*np.log(10), np.log(10)]
        xticks = [-1, 0, 1]
        xticklabels = [-1, 0, 1]
        xlabel = 'speed'
    
    
    #########
    keys_upwind = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['on', 'upwind'])
    ax = fig.add_subplot(131)
    yticks_neg = [-9,-8,-7,-6,-5,-4,-3,-2,-1,0]
    yticks_neg = -1*np.log(-1*np.array(yticks_neg)+1)
    yticks_pos = [1,2,3,4,5,6,7,8,9]
    yticks_pos = np.log(np.array(yticks_pos)+1)
    yticks = yticks_neg.tolist()
    yticks.extend(yticks_pos.tolist())
    ytick_labels = [-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9]
    
    ax.set_aspect('auto')
    ax.set_ylim(ylim[0], ylim[1])
    
    plot_heading_time_to_odor_on_ax(ax, dataset, keys=keys_upwind, attribute=attribute, column=column, xlim=xlim, relative_to=relative_to, threshold_odor=threshold_odor)
    ax.vlines(-1.15, -10, 10, color='black') # np.pi - np.arctan2( mean(y vel during cast), visual wind speed )
    ax.vlines(1.15, -10, 10, color='black')
    ax.set_title('Ground moving upwind')
    
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel(xlabel)
    
    if relative_to == 'entry':
        ax.set_ylabel('time relative to odor entry, sec')
    else:
        ax.set_ylabel('time relative to odor exit, sec')
    
    ax.set_aspect('auto')

    ##########
    keys_downwind = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['on', 'downwind'])
    ax = fig.add_subplot(132)
    keys_to_show = ['20120926_174720_6196']
    
    ax.set_aspect('auto')
    ax.set_ylim(ylim[0], ylim[1])
    
    plot_heading_time_to_odor_on_ax(ax, dataset, keys=keys_downwind, attribute=attribute, column=column, xlim=xlim, relative_to=relative_to, threshold_odor=threshold_odor)
    ax.vlines(-1.98, -10, 10, color='black')
    ax.vlines(1.98, -10, 10, color='black')
    ax.set_title('Ground moving downwind')
    
    fpl.adjust_spines(ax, ['bottom'], xticks=xticks, yticks=yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_aspect('auto')
    
    ##########
    keys = fad.get_keys_with_attr(dataset, ['odor_stimulus'], ['on'])
    ax = fig.add_subplot(133)
    keys_to_show = ['20120926_174720_6196']
    
    ax.set_aspect('auto')
    ax.set_ylim(ylim[0], ylim[1])
    
    print 'plotting auto adjusted floor plot, could take some time'
    plot_heading_time_to_odor_on_ax(ax, dataset, keys=keys_downwind, attribute=attribute, column=column, xlim=xlim, relative_to=relative_to, threshold_odor=threshold_odor, auto_velocity_adjustment=True)
    ax.vlines(-np.pi/2., -10, 5, color='black')
    ax.vlines(np.pi/2., -10, 5, color='black')
    ax.set_title('Heading relative to ground motion')
    
    fpl.adjust_spines(ax, ['bottom'], xticks=xticks, yticks=yticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_aspect('auto')
    
    
    path = config.path
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    figname = attribute + '_vs_time_' + 'relative_to_' + relative_to + '_odor.pdf'
    pdf_name_with_path = os.path.join(save_figure_path, figname)
    
    print 'SAVING TO: ', pdf_name_with_path
    
    fig.savefig(pdf_name_with_path, format='pdf')


###################################################################################################################################
# Angular velocity stuff
###################################################################################################################################

def plot_angular_velocity_for_trajec_on_ax(ax, trajec, threshold_odor=10, attribute='heading_smooth_diff', column=0, relative_to='entry', framerange=None):
        attribute, relative_time = get_attribute_and_time_relative_to_odor(trajec, threshold_odor=threshold_odor, attribute=attribute, column=column, relative_to=relative_to, framerange=framerange)
        
        heading, relative_time = get_attribute_and_time_relative_to_odor(trajec, threshold_odor=threshold_odor, attribute='heading_smooth', column=column, relative_to=relative_to, framerange=framerange)
    
        if len(relative_time) < 10:
            return None
            
        time_chunks, attribute_chunks, breakpoints = flymath.get_continuous_chunks(relative_time, attribute, jump=0.02)
        time_chunks, heading_chunks, breakpoints = flymath.get_continuous_chunks(relative_time, heading, jump=0.02)
        
        for i, relative_time in enumerate(time_chunks):
            attribute_vals = np.array(attribute_chunks[i])
            relative_time = np.array(relative_time)
            heading = np.array(heading_chunks[i])
            
            if relative_to == 'exit':
                # handle time after odor
                indices = np.where(relative_time>0)[0]
                attribute_vals_after = attribute_vals[indices]
                time_after = relative_time[indices]
                heading_after = heading[indices]
                
                angular_vel_after = attribute_vals_after
                if len(angular_vel_after) < 150:
                    continue
                elif np.max(np.abs(angular_vel_after[40:130])) < 300*np.pi/180.:
                    continue
                else:
                    index_at_max_angular_velocity = np.where( np.abs(angular_vel_after[40:130]) >= 300*np.pi/180.)[0][0] + 40
                    time_relative_to_max_angular_velocity = time_after - time_after[index_at_max_angular_velocity]
                
                if 0:
                    if np.abs(heading_after[0]) < 30*np.pi/180.: 
                        pass
                    else:
                        continue
                
                if 0:
                    # handle time before odor
                    indices = np.where(time_relative_to_max_angular_velocity<0)[0]
                    angular_vel_after_odor_prior_max = angular_vel_after[indices]
                    time_prior = time_relative_to_max_angular_velocity[indices]
                    time_prior = -1*np.log(-1*time_prior + 1)
                    # handle time after odor
                    indices = np.where(time_relative_to_max_angular_velocity>0)[0]
                    angular_vel_after_odor_after_max = angular_vel_after[indices]
                    time_after = time_relative_to_max_angular_velocity[indices]
                    time_after = np.log(time_after + 1)
                    
                    # merge them
                    time_merged = np.hstack((time_prior, time_after))
                    attribute_vals_merged = np.hstack((angular_vel_after_odor_prior_max, angular_vel_after_odor_after_max))
                if 1:
                    try:
                        ax.plot(angular_vel_after, time_relative_to_max_angular_velocity, color='black', alpha=0.2)
                    except:
                        print 'no dice'
                        
                if 0:
                    print trajec.key
                    heading, relative_time, breakpoints = flymath.get_continuous_chunks(heading, relative_time, jump=2)
                    for i in range(len(relative_time)):
                        if len(heading[i]) > 10:
                            print np.max(np.abs(flymath.diffa(heading[i]))/100.)
                            fpl.colorline(ax, heading[i], relative_time[i], np.abs(flymath.diffa(heading[i]))*100., alpha=0.3, norm=[0,10])
                
            if relative_to == 'entry':
                # handle time after odor
                indices = np.where(relative_time>0)[0]
                attribute_vals_after = attribute_vals[indices]
                time_after = relative_time[indices]
                heading_after = heading[indices]
                
                angular_vel_after = attribute_vals_after
                if len(angular_vel_after) < 71:
                    continue
                elif np.max(np.abs(angular_vel_after[0:20])) < 300*np.pi/180.:
                    continue
                else:
                    index_at_max_angular_velocity = np.where( np.abs(angular_vel_after[0:20]) >= 300*np.pi/180.)[0][0]
                    time_relative_to_max_angular_velocity = time_after - time_after[index_at_max_angular_velocity]
                
                if 0:  
                    if np.abs(heading_after[0]) > 60*np.pi/180 and np.abs(heading_after[0]) < 110*np.pi/180: 
                        pass
                    else:
                        continue
                    
                if 0:
                    # handle time before odor
                    indices = np.where(time_relative_to_max_angular_velocity<0)[0]
                    angular_vel_after_odor_prior_max = angular_vel_after[indices]
                    time_prior = time_relative_to_max_angular_velocity[indices]
                    time_prior = -1*np.log(-1*time_prior + 1)
                    # handle time after odor
                    indices = np.where(time_relative_to_max_angular_velocity>0)[0]
                    angular_vel_after_odor_after_max = angular_vel_after[indices]
                    time_after = time_relative_to_max_angular_velocity[indices]
                    time_after = np.log(time_after + 1)
                    
                    # merge them
                    time_merged = np.hstack((time_prior, time_after))
                    attribute_vals_merged = np.hstack((angular_vel_after_odor_prior_max, angular_vel_after_odor_after_max))
                
                    print trajec.key
                    ax.plot(attribute_vals_merged, time_merged, color='black', alpha=0.2)
            
                if 1:
                    try:
                        ax.plot(angular_vel_after, time_relative_to_max_angular_velocity, color='black', alpha=0.2)
                    except:
                        print 'no dice'
                
                if 0:
                    print trajec.key
                    #relative_time, heading = flymath.remove_discontinuities(relative_time, heading, 2)
                    heading, relative_time, breakpoints = flymath.get_continuous_chunks(heading, relative_time, jump=2)
                    for i in range(len(relative_time)):
                        if len(heading[i]) > 10:
                            print np.max(np.abs(flymath.diffa(heading[i]))/100.)
                            fpl.colorline(ax, heading[i], relative_time[i], np.abs(flymath.diffa(heading[i]))*100., alpha=0.3, norm=[0,10])
                    
          
def plot_angular_velocity_for_dataset_on_ax(ax, dataset, keys=None, attribute='heading_smooth_diff', column=0, xlim=[-1,1], relative_to='entry', threshold_odor=10):

    for key in keys:
        trajec = dataset.trajecs[key]
        plot_angular_velocity_for_trajec_on_ax(ax, trajec, threshold_odor=threshold_odor, attribute=attribute, column=column, relative_to=relative_to, framerange=None)
            
def plot_angular_velocity_relative_to(dataset, config, threshold_odor=10, attribute='heading_smooth_diff', column=0, relative_to='entry', framerange=None):
    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    if attribute == 'heading_smooth_diff':
        print attribute
        ylim = [-.3, .3]
        if 1:
            xlim = [-5000*np.pi/180, 5000*np.pi/180]
            xticks = [-5000*np.pi/180, 0, 5000*np.pi/180]
            xticklabels = [-5000, 0, 5000]
            xlabel = 'angular velocity, deg/s'
        if 0:
            xlim = [-np.pi, np.pi]
            xticks = [-np.pi, np.pi]
            xticklabels = [-np.pi, np.pi]
            xlabel = 'heading'
    else:
        print 'Choose a valid attribute!'
        
    yticks = [-.3,0,1]
    ytick_labels = [-.3,0,1]
    
    if 1:
        keys_odor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'on')
        print 'n keys with odor: ', len(keys_odor)
        ax = fig.add_subplot(121)
        keys_to_show = ['20120926_174720_6196']
        
        ax.set_aspect('auto')
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(xlim[0], xlim[1])
        
        plot_angular_velocity_for_dataset_on_ax(ax, dataset, keys=keys_odor, attribute=attribute, column=column, xlim=xlim, relative_to='entry', threshold_odor=threshold_odor)
        ax.vlines(-300*np.pi/180., -10, 5, color='red')
        ax.vlines(300*np.pi/180., -10, 5, color='red')
        ax.hlines(0.05, -5000, 5000, color='purple')
        ax.hlines(0.1, -5000, 5000, color='blue')
        ax.hlines(0.2, -5000, 5000, color='green')
        ax.hlines(0.3, -5000, 5000, color='darkgoldenrod')
        ax.hlines(0.4, -5000, 5000, color='orange')
        ax.hlines(0.5, -5000, 5000, color='red')
        ax.set_title('relative to odor entry')
        
        fpl.adjust_spines(ax, ['bottom', 'left'], xticks=xticks, yticks=yticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(xlabel)
        ax.set_aspect('auto')
        ax.set_ylabel('time relative to first saccade after odor experience')

    ##########
    if 1:
        keys_odor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'on')
        print 'n keys with odor: ', len(keys_odor)
        ax = fig.add_subplot(122)
        keys_to_show = ['20120926_174720_6196']
        
        ax.set_aspect('auto')
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(xlim[0], xlim[1])
        
        plot_angular_velocity_for_dataset_on_ax(ax, dataset, keys=keys_odor, attribute=attribute, column=column, xlim=xlim, relative_to='exit', threshold_odor=threshold_odor)
        ax.vlines(-300*np.pi/180., -10, 5, color='red')
        ax.vlines(300*np.pi/180., -10, 5, color='red')
        ax.hlines(0.05, -5000, 5000, color='purple')
        ax.hlines(0.1, -5000, 5000, color='blue')
        ax.hlines(0.2, -5000, 5000, color='green')
        ax.hlines(0.3, -5000, 5000, color='darkgoldenrod')
        ax.hlines(0.4, -5000, 5000, color='orange')
        ax.hlines(0.5, -5000, 5000, color='red')
        ax.set_title('relative to odor exit')
        
        fpl.adjust_spines(ax, ['bottom', 'left'], xticks=xticks, yticks=yticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(xlabel)
        ax.set_aspect('auto')
        
    path = config.path
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    figname = attribute + '_vs_time_' + 'relative_to_' + relative_to + '_odor' + '_threshold_' + str(threshold_odor) + '.pdf'
    pdf_name_with_path = os.path.join(save_figure_path, figname)
    
    print 'SAVING TO: ', pdf_name_with_path
    
    fig.savefig(pdf_name_with_path, format='pdf')
    
#############################################################################################################################
# Saccades
###################


def get_saccades_relative_to_odor_for_trajec(trajec, threshold_odor=10, relative_to='entry', sac_addition=0):
    if np.max(trajec.odor) < threshold_odor:
        return [], [], []
        
    frame_chunks_prior_odor, frame_chunks_after_odor = get_frame_chunks_relative_to_odor(trajec, threshold_odor=threshold_odor)
    
    if relative_to == 'entry':
        chunks = frame_chunks_prior_odor
    elif relative_to == 'exit':
        chunks = frame_chunks_after_odor
        
    if relative_to == 'entry':
        frame_range = [0,20]
    elif relative_to == 'exit':
        frame_range = [40,80]
        
    headings_prior = []
    amplitudes = []
    speeds = []
        
    n = 0
    for frame_rel, frame_chunk in chunks.items():
        if n == 0:
            pass
            
        for sac in trajec.saccades:
            if sac[0] > frame_rel:
                if flymath.in_range(sac[0]-frame_rel, frame_range):
                    if sac[-1] + sac_addition > trajec.length-1:
                        break
                    headings_prior.append(trajec.heading_smooth[sac[0]])
                    a = flymath.fix_angular_rollover(trajec.heading_smooth[sac[0]+sac_addition] - trajec.heading_smooth[sac[0]])
                    amplitudes.append( a )
                    speeds.append(trajec.speed_xy[sac[0]])
                    break
                    
    return headings_prior, amplitudes, speeds
    

def get_headings_relative_to_odor_for_trajec(trajec, threshold_odor=10, relative_to='entry'):
    if np.max(trajec.odor) < threshold_odor:
        return [], [], [], []
        
    frame_chunks_prior_odor, frame_chunks_after_odor = get_frame_chunks_relative_to_odor(trajec, threshold_odor=threshold_odor)
    
    if relative_to == 'entry':
        chunks = frame_chunks_prior_odor
    elif relative_to == 'exit':
        chunks = frame_chunks_after_odor
        
    if relative_to == 'entry':
        frame_range = [0,20]
    elif relative_to == 'exit':
        frame_range = [40,80]
        
    headings_prior = []
    headings_after = []
    headings_after2 = []
    headings_after3 = []
        
    n = 0
    for frame_rel, frame_chunk in chunks.items():
        if n == 0:
            pass
            
        for sac in trajec.saccades:
            if sac[0] > frame_rel:
                if flymath.in_range(sac[0]-frame_rel, frame_range):
                    if sac[-1]+40 > trajec.length:
                        break
                    headings_prior.append(trajec.heading_smooth[sac[0]])
                    headings_after.append(trajec.heading_smooth[sac[-1]])
                    headings_after2.append(trajec.heading_smooth[sac[-1]+20])
                    headings_after3.append(trajec.heading_smooth[sac[-1]+30])
                    break
                    
    return headings_prior, headings_after, headings_after2, headings_after3
    
    
def get_headings_relative_to_odor_for_dataset(dataset, keys=None, threshold_odor=10, relative_to='entry', sac_addition=0):
    if keys is None:
        keys = dataset.trajecs.keys()
        
    headings_prior = []
    headings_after = []
    headings_after2 = []
    headings_after3 = []
    
    for key in keys:
        trajec = dataset.trajecs[key]
        h0, h1, h2, h3 = get_headings_relative_to_odor_for_trajec(trajec, threshold_odor=threshold_odor, relative_to=relative_to)
        headings_prior.extend(h0)
        headings_after.extend(h1)
        headings_after2.extend(h2)
        headings_after3.extend(h3)
        
    return headings_prior, headings_after, headings_after2, headings_after3
    
    
    
def get_saccades_relative_to_odor_for_dataset(dataset, keys=None, threshold_odor=10, relative_to='entry', sac_addition=0):
    if keys is None:
        keys = dataset.trajecs.keys()
        
    headings_prior = []
    amplitudes = []
    speeds = []
    
    for key in keys:
        trajec = dataset.trajecs[key]
        h, a, s = get_saccades_relative_to_odor_for_trajec(trajec, threshold_odor=threshold_odor, relative_to=relative_to, sac_addition=sac_addition)
        headings_prior.extend(h)
        amplitudes.extend(a)
        speeds.extend(s)
        
    return headings_prior, amplitudes, speeds
    
def plot_saccades_relative_to_odor(dataset, dataset_nowind, dataset_moving, config, threshold_odor=10):
    
    keys_odor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'on')
    keys_noodor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'none')
    keys_after = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'afterodor')
    keys_noodor.extend(keys_after)
    
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'saccade_angle_relative_to_odor.pdf')
    pp = PdfPages(pdf_name_with_path)
    
    markersize = 2
    ticks = [-180,-90,0,90,180]
    binsx = np.linspace(-180,180,50)
    binsy = np.linspace(-180,180,50)
    
    ### odor on
    fig = plt.figure(figsize=(8,5))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    headings_prior, amplitudes, speeds = get_saccades_relative_to_odor_for_dataset(dataset, keys=keys_odor, threshold_odor=threshold_odor, relative_to='entry')
    amplitudes = flymath.fix_angular_rollover(amplitudes)
    ax1.plot(np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, '.', color='blue', markersize=markersize) 
    #fpl.histogram2d(ax1, np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, bins=(binsx,binsy), logcolorscale=True)
    #fpl.scatter(ax1, np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, color=np.array(speeds), colornorm=[0,0.6], radius=2)
    ax1.set_xlim(-180,180)
    ax1.set_ylim(-180,180)
    ax1.set_aspect('equal')
    fpl.adjust_spines(ax1, ['left', 'bottom'], xticks=ticks, yticks=ticks)
    ax1.set_xlabel('heading prior to saccade')
    ax1.set_ylabel('angle of saccade')
    ax1.set_title('saccades 0-0.5 s\nafter entering odor')
    
    # plot line
    ax1.plot(ticks, -1*np.array(ticks), zorder=-10, color='black')
    
    headings_prior, amplitudes, speeds = get_saccades_relative_to_odor_for_dataset(dataset, keys=keys_odor, threshold_odor=threshold_odor, relative_to='exit')
    amplitudes = flymath.fix_angular_rollover(amplitudes)
    ax2.plot(np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, '.', color='blue', markersize=markersize) 
    #fpl.histogram2d(ax2, np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, bins=(binsx,binsy), logcolorscale=True)
    #fpl.scatter(ax2, np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, color=np.array(speeds), colornorm=[0,0.6], radius=2)
    ax2.set_xlim(-180,180)
    ax2.set_ylim(-180,180)
    ax2.set_aspect('equal')
    fpl.adjust_spines(ax2, ['bottom'], xticks=ticks, yticks=ticks)
    ax2.set_title('saccades 0.5-1.3 s\nafter leaving odor')
    
    pp.savefig()
    plt.close('all')
    
    ### moving
    fig = plt.figure(figsize=(8,5))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    keys_upwind = fad.get_keys_with_attr(dataset_moving, ['odor_stimulus', 'visual_stimulus'], ['on', 'upwind'])
    headings_prior, amplitudes, speeds = get_saccades_relative_to_odor_for_dataset(dataset_moving, keys=keys_upwind, threshold_odor=threshold_odor, relative_to='entry')
    amplitudes = flymath.fix_angular_rollover(amplitudes)
    ax1.plot(np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, '.', color='blue', markersize=markersize) 
    #fpl.histogram2d(ax1, np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, bins=(binsx,binsy), logcolorscale=True)
    ax1.set_xlim(-180,180)
    ax1.set_ylim(-180,180)
    ax1.set_aspect('equal')
    fpl.adjust_spines(ax1, ['left', 'bottom'], xticks=ticks, yticks=ticks)
    ax1.set_xlabel('heading prior to saccade')
    ax1.set_ylabel('angle of saccade')
    ax1.set_title('saccades 0-0.5 s\nafter entering odor')
    
    # plot line
    ax1.plot(ticks, -1*np.array(ticks), zorder=-10, color='black')
    
    keys_downwind = fad.get_keys_with_attr(dataset_moving, ['odor_stimulus', 'visual_stimulus'], ['on', 'downwind'])
    headings_prior, amplitudes, speeds = get_saccades_relative_to_odor_for_dataset(dataset_moving, keys=keys_downwind, threshold_odor=threshold_odor, relative_to='entry')
    amplitudes = flymath.fix_angular_rollover(amplitudes)
    ax2.plot(np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, '.', color='blue', markersize=markersize) 
    #fpl.histogram2d(ax2, np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, bins=(binsx,binsy), logcolorscale=True)
    ax2.set_xlim(-180,180)
    ax2.set_ylim(-180,180)
    ax2.set_aspect('equal')
    fpl.adjust_spines(ax2, ['bottom'], xticks=ticks, yticks=ticks)
    ax2.set_title('saccades 0.5-1.3 s\nafter leaving odor')
    
    pp.savefig()
    plt.close('all')
    
    ### no wind
    fig = plt.figure(figsize=(8,5))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    headings_prior, amplitudes, speeds = get_saccades_relative_to_odor_for_dataset(dataset_nowind, keys=dataset_nowind.trajecs.keys(), threshold_odor=threshold_odor, relative_to='entry')
    amplitudes = flymath.fix_angular_rollover(amplitudes)
    ax1.plot(np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, '.', color='blue', markersize=markersize) 
    #fpl.histogram2d(ax1, np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, bins=(binsx,binsy), logcolorscale=True)
    ax1.set_xlim(-180,180)
    ax1.set_ylim(-180,180)
    ax1.set_aspect('equal')
    fpl.adjust_spines(ax1, ['left', 'bottom'], xticks=ticks, yticks=ticks)
    ax1.set_xlabel('heading prior to saccade')
    ax1.set_ylabel('angle of saccade')
    ax1.set_title('saccades 0-0.5 s\nafter entering odor')
    
    # plot line
    ax1.plot(ticks, -1*np.array(ticks), zorder=-10, color='black')
    
    headings_prior, amplitudes, speeds = get_saccades_relative_to_odor_for_dataset(dataset_nowind, keys=dataset_nowind.trajecs.keys(), threshold_odor=threshold_odor, relative_to='exit')
    amplitudes = flymath.fix_angular_rollover(amplitudes)
    ax2.plot(np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, '.', color='blue', markersize=markersize) 
    #fpl.histogram2d(ax2, np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, bins=(binsx,binsy), logcolorscale=True)
    ax2.set_xlim(-180,180)
    ax2.set_ylim(-180,180)
    ax2.set_aspect('equal')
    fpl.adjust_spines(ax2, ['bottom'], xticks=ticks, yticks=ticks)
    ax2.set_title('saccades 0.5-1.3 s\nafter leaving odor')
    
    pp.savefig()
    plt.close('all')
    
    pp.close()
    
    
    
def plot_saccades_relative_to_odor_with_addition(dataset, config, threshold_odor=10):
    
    keys_odor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'on')
    keys_noodor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'none')
    keys_after = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'afterodor')
    keys_noodor.extend(keys_after)
    
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'saccade_angle_relative_to_odor.pdf')
    
    markersize = 2
    ticks = [-180,-90,0,90,180]
    binsx = np.linspace(-180,180,50)
    binsy = np.linspace(-180,180,50)
    
    
    sac_additions = [5, 10, 20, 30, 40, 50]
    colors = ['purple', 'blue', 'green', 'darkgoldenrod', 'orange', 'red']
    
    fig = plt.figure(figsize=(10,3))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, hspace=0.2, wspace=0.2)
    
    
    
    for i, sac_addition in enumerate(sac_additions):
        ### odor on
        
        ax = fig.add_subplot(1,len(sac_additions),i+1)
        
        headings_prior, amplitudes, speeds = get_saccades_relative_to_odor_for_dataset(dataset, keys=keys_odor, threshold_odor=threshold_odor, relative_to='entry', sac_addition=sac_addition)
        amplitudes = flymath.fix_angular_rollover(amplitudes)
        ax.plot(np.array(headings_prior)*180/np.pi, np.array(amplitudes)*180/np.pi, '.', color=colors[i], markersize=markersize) 
        ax.plot(ticks, -1*np.array(ticks), zorder=-10, color='black')
        ax.set_xlim(-180,180)
        ax.set_ylim(-180,180)
        ax.set_aspect('equal')
        if i==0:
            fpl.adjust_spines(ax, ['left', 'bottom'], xticks=ticks, yticks=ticks)
            ax.set_ylabel('angle of turn')
        else:
            fpl.adjust_spines(ax, ['bottom'], xticks=ticks, yticks=ticks)
        ax.set_xlabel('heading prior to turn')
        title_text = 'after ' + str(sac_addition*10) + ' ms'
        ax.set_title(title_text)
        
    fig.suptitle('time course of turn vs. initial heading, relative to turn initiation\nfor turns made within 200 ms after entering the odor plume')
        
    fig.savefig(pdf_name_with_path, format='pdf')
    
    
    
def heading_after_saccade(dataset, config, threshold_odor=10):
    
    keys_odor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'on')
    
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'heading_after_saccade_after_odor.pdf')
    
    markersize = 2
    ticks = [-180,-90,0,90,180]
    binsx = np.linspace(-180,180,50)
    binsy = np.linspace(-180,180,50)
    
    ### odor on
    fig = plt.figure(figsize=(8,5))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    ax = fig.add_subplot(111)
    
    headings_prior, headings_after, headings_after2, headings_after3 = get_headings_relative_to_odor_for_dataset(dataset, keys=keys_odor, threshold_odor=threshold_odor, relative_to='entry')
    
    headings_prior = np.array(headings_prior)*180/np.pi
    headings_after = np.array(headings_after)*180/np.pi
    headings_after2 = np.array(headings_after2)*180/np.pi
    headings_after3 = np.array(headings_after3)*180/np.pi
    
    #fpl.histogram(ax, [headings_prior, headings_after, headings_after2, headings_after3], bins=30, colors=['blue', 'green', 'orange', 'red'], normed=True)
    
    ax.plot(headings_prior, headings_after2, '.')
    
    ax.set_xlim(-180,180)
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=ticks)


    fig.savefig(pdf_name_with_path, format='pdf')
    
            
####################################################################

def calc_heading(velocities, smooth=False):
    if smooth is False:
        heading_smooth = np.arctan2(velocities[:,1], velocities[:,0])
        # flip
        ip = np.where(heading_smooth>0)
        im = np.where(heading_smooth<=0)
        heading_smooth[ip] -= np.pi
        heading_smooth[im] += np.pi
        return heading_smooth # this is not actually smooth
        
        
    heading_norollover = floris_math.remove_angular_rollover(np.arctan2(velocities[:,1], velocities[:,0]), 3)
    ## kalman
    
    data = heading_norollover.reshape([len(heading_norollover),1])
    ss = 3 # state size
    os = 1 # observation size
    F = np.array([   [1,1,0], # process update
                     [0,1,1],
                     [0,0,1]],
                    dtype=np.float)
    H = np.array([   [1,0,0]], # observation matrix
                    dtype=np.float)
    Q = np.eye(ss) # process noise
    Q[0,0] = .01
    Q[1,1] = .01
    Q[2,2] = .01
    R = 1*np.eye(os) # observation noise
    
    initx = np.array([data[0,0], data[1,0]-data[0,0], 0], dtype=np.float)
    initv = 0*np.eye(ss)
    xsmooth,Vsmooth = kalman_math.kalman_smoother(data, F, H, Q, R, initx, initv, plot=False)

    heading_norollover_smooth = xsmooth[:,0]
    heading_smooth_diff = xsmooth[:,1]*100.
    
    heading = floris_math.fix_angular_rollover(heading_norollover)
    heading_smooth = floris_math.fix_angular_rollover(heading_norollover_smooth)
    
    # flip
    ip = np.where(heading_smooth>0)
    im = np.where(heading_smooth<=0)
    heading_smooth[ip] -= np.pi
    heading_smooth[im] += np.pi
    
        
    return heading_smooth


def plot_time_since_odor(config, dataset, keys=None):

    fig = plt.figure(figsize=(10,4))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    
    
    
    timeranges = [[0,0.1], [0.1, 0.4], [0.4,1], [1,2], [2,5], [5,10]]
    
    if keys is None:
        keys = dataset.trajecs.keys()
    
    for n, timerange in enumerate(timeranges):
        print timerange
        time_since_odor = []
        heading_after_saccade = []
        heading_before_saccade = []
    
        for key in keys:
            trajec = dataset.trajecs[key]
            frames_where_odor = np.where(trajec.odor>5)[0]
            for sac in trajec.saccades:
            
                tmp = sac[0] - np.array(frames_where_odor)
                try:
                    tso = tmp[np.where(tmp>0)[0][-1]]/trajec.fps
                except:
                    tso = 10000
                    
                if tso < timerange[1] and tso > timerange[0]:
                    time_since_odor.append(tso)
                    heading_after_saccade.append(trajec.heading_smooth[sac[-1]])
                    heading_before_saccade.append(trajec.heading_smooth[sac[0]])
            
            
    
    
        axscatter = fig.add_subplot(2,int(len(timeranges)/2.),n+1)
        binsx = np.linspace(-np.pi, np.pi, 50)
        binsy = np.linspace(-np.pi, np.pi, 50)
        
        if len(heading_after_saccade) > 0:
            fpl.histogram2d(axscatter, heading_before_saccade, heading_after_saccade, bins=(binsx, binsy), logcolorscale=True)
        axscatter.set_xlim(-np.pi, np.pi)
        axscatter.set_ylim(-np.pi, np.pi)
        
        ticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
        ticklabels = [-180, -90, 0, 90, 180]
        axscatter.set_aspect('equal')
        
        if n == 0:
            fpl.adjust_spines(axscatter,['left', 'bottom'], xticks=ticks, yticks=ticks)
            axscatter.set_xlabel('heading before saccade')
            axscatter.set_ylabel('heading after saccade')
            axscatter.set_xticklabels(ticklabels)
            axscatter.set_yticklabels(ticklabels)
        else:
            fpl.adjust_spines(axscatter,[], xticks=ticks, yticks=ticks)
            
        title_str = 'time since odor:\n' + str(timerange[0]) + ':' + str(timerange[1]) + ' sec'
        axscatter.set_title(title_str)
                
            
    
    path = config.path
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'saccade_timings.pdf')
    
    print 'SAVING TO: ', pdf_name_with_path
    
    fig.savefig(pdf_name_with_path, format='pdf')
    
    
def classify_saccade(trajec, sac):

    radius = 25
    
    heading_prior = trajec.heading_smooth[sac[0]]*180/np.pi
    heading_after = trajec.heading_smooth[sac[-1]]*180/np.pi
    saccade = np.array([heading_prior, heading_after])
    classification = None
    
    # distance to point
    surge_cast_p = np.array([0,90])
    surge_cast_n = np.array([0,-90])
    cast_surge_p = np.array([90,0])
    cast_surge_n = np.array([-90,0])
    cast_cast_p = np.array([90,-90])
    cast_cast_n = np.array([-90,90])
    
    classifications = {   'surge_cast_p': surge_cast_p, 
                        'surge_cast_n': surge_cast_n, 
                        'cast_surge_p': cast_surge_p, 
                        'cast_surge_n': cast_surge_n, 
                        'cast_cast_p':  cast_cast_p, 
                        'cast_cast_n':  cast_cast_n,
                    }
                    
    for key, item in classifications.items():
        if np.linalg.norm(saccade - item) < radius:
            return key
    
    return None
    

def plot_time_and_saccade_data(config, dataset, keys=None):

    fig = plt.figure(figsize=(10,4))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    
    classifications = {}
    
    
    if keys is None:
        keys = dataset.trajecs.keys()
        
    n = 0
    
    for key in keys:
        print key
        trajec = dataset.trajecs[key]
        frames_where_odor = np.where(trajec.odor>10)[0]
        if len(trajec.saccades) == 0:
            continue
        for sac in trajec.saccades:
        
            tmp = sac[0] - np.array(frames_where_odor)
            try:
                tso = tmp[np.where(tmp>0)[0][-1]]/trajec.fps
            except:
                tso = 10000
                
        if tso < 5:
            n += 1
            classification = classify_saccade(trajec, sac)
            print classification
            if classification is not None:
                if classifications.has_key(classification):
                    classifications[classification].append(tso)
                else:
                    classifications.setdefault(classification, [tso])
                
    behaviors = {'cast_surge': [], 'surge_cast': [], 'cast_cast': []}
    colors = {'cast_surge': 'red', 'surge_cast': 'green', 'cast_cast': 'black'}
    for behavior, data in behaviors.items():
        behaviors[behavior].extend(classifications[behavior+'_p'])
        behaviors[behavior].extend(classifications[behavior+'_n'])
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for behavior, data in behaviors.items():
        print behavior, ': ', len(data)
    
    print 'n saccades: ', n
    
    fpl.histogram(ax, behaviors.values(), bins=40, bin_width_ratio=0.8, colors=colors.values(), edgecolor='none', bar_alpha=1, curve_fill_alpha=0.4, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False)           
            
            
            
def plot_time_since_odor_and_heading_on_ax(ax, config, dataset, keys, keys_to_show=[], velocity_adjustment=0, auto_velocity_adjustment=False):

    odor_threshold = 10

    if keys is None:
        keys = dataset.trajecs.keys()
        
    headings = []
    time_since_odor = []
    
    y_velocities = []
    
        
    for key in keys:
        trajec = dataset.trajecs[key]
        frames_where_odor = np.where(trajec.odor>odor_threshold)[0]
        
        if auto_velocity_adjustment:
            if trajec.visual_stimulus == 'upwind':
                velocity_adjustment = 0.08
            elif trajec.visual_stimulus == 'downwind':
                velocity_adjustment = -0.08
            else:
                velocity_adjustment = 0
                
        
        if velocity_adjustment != 0:
            velocities = copy.copy(trajec.velocities[:,0:2])
            velocities[:,0] += velocity_adjustment
            heading_smooth = calc_heading(velocities)
        else:
            heading_smooth = trajec.heading_smooth
        
        headings_for_trajec = []
        time_since_odor_for_trajec = []
        
        for f in range(trajec.length):
            tmp = f - np.array(frames_where_odor)
            try:
                tso = tmp[np.where(tmp>0)[0][-1]]/trajec.fps
            except:
                tso = 10000
                
            if tso < 20 and tso > 0.01 and np.abs(trajec.positions[f,2]) < 0.13:
                headings.append(heading_smooth[f])
                time_since_odor.append(tso)
                
            if np.abs(np.abs(heading_smooth[f])*180/np.pi - 90) < 10:
                if not np.isnan(trajec.velocities[f,1]):
                    y_velocities.append(trajec.velocities[f,1])
                    
            time_since_odor_for_trajec.append(tso)
            headings_for_trajec.append(heading_smooth[f])
                    
        if key in keys_to_show:
            time_since_odor_for_trajec, headings_for_trajec = flymath.remove_discontinuities(time_since_odor_for_trajec, headings_for_trajec, 3)
            headings_for_trajec, time_since_odor_for_trajec = flymath.remove_discontinuities(headings_for_trajec, time_since_odor_for_trajec, .25)
            ax.plot(headings_for_trajec, np.log(np.array(time_since_odor_for_trajec)+1), '-', zorder=10, color='black')
            
            
    print 'Y velocities: ', np.mean(np.abs(y_velocities))
            
    binsx = np.linspace(-np.pi,np.pi,100)
    binsy = np.linspace(0,np.log(10),100)
    fpl.histogram2d(ax, headings, np.log(np.array(time_since_odor)+1), bins=(binsx, binsy), logcolorscale=True)            
    
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0,np.log(10))
    
    ax.set_aspect('auto')
    

def plot_time_since_odor_and_heading(config, dataset):
    
    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    keys_noodor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'none')
    ax = fig.add_subplot(121)
    plot_time_since_odor_and_heading_on_ax(ax, config, dataset, keys_noodor)
    ax.vlines(-np.pi/2., 0, 10, color='black')
    ax.vlines(np.pi/2., 0, 10, color='black')
    ax.set_title('odor off')
    
    yticks = np.array([0,1,2,3,4,5,6,7,8,9])
    yticks = np.log(yticks+1)
    yticklabels = yticks.tolist()
    
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=[-np.pi, -np.pi/2., 0, np.pi/2., np.pi], yticks=yticks)
    ax.set_xticklabels([-180,-90,0,90,180])
    ax.set_yticklabels([0,1,2,3,4,5,6,7,8,9])
    ax.set_xlabel('heading')
    ax.set_ylabel('time since odor, sec')

    keys_odor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'on')
    ax = fig.add_subplot(122)
    keys_to_show = ['20120926_174720_6196']
    plot_time_since_odor_and_heading_on_ax(ax, config, dataset, keys_odor, keys_to_show=keys_to_show)
    ax.vlines(-np.pi/2., 0, 10, color='black')
    ax.vlines(np.pi/2., 0, 10, color='black')
    ax.set_title('odor on')
    fpl.adjust_spines(ax, ['bottom'], xticks=[-np.pi, -np.pi/2., 0, np.pi/2., np.pi], yticks=yticks)
    ax.set_xticklabels([-180,-90,0,90,180])
    ax.set_xlabel('heading')
    
    path = config.path
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'heading_vs_time_since_odor.pdf')
    
    print 'SAVING TO: ', pdf_name_with_path
    
    fig.savefig(pdf_name_with_path, format='pdf')
    
    
def plot_time_since_odor_and_heading_moving_floor(config, dataset):
    
    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    keys_upwind = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['on', 'upwind'])
    ax = fig.add_subplot(131)
    plot_time_since_odor_and_heading_on_ax(ax, config, dataset, keys_upwind)
    ax.vlines(-1.15, 0, 10, color='black') # np.pi - np.arctan2( mean(y vel during cast), visual wind speed )
    ax.vlines(1.15, 0, 10, color='black')
    ax.set_title('upwind visual motion')
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=[-np.pi, -np.pi/2., 0, np.pi/2., np.pi], yticks=[0,1,2,3,4,5,6,7,8,9,10])
    ax.set_xticklabels([-180,-90,0,90,180])
    ax.set_xlabel('heading')
    ax.set_ylabel('time since odor, sec')

    keys_downwind = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['on', 'downwind'])
    ax = fig.add_subplot(132)
    plot_time_since_odor_and_heading_on_ax(ax, config, dataset, keys_downwind)
    ax.vlines(-1.98, 0, 10, color='black')
    ax.vlines(1.98, 0, 10, color='black')
    ax.set_title('downwind visual motion')
    fpl.adjust_spines(ax, ['bottom'], xticks=[-np.pi, -np.pi/2., 0, np.pi/2., np.pi], yticks=[0,1,2,3,4,5,6,7,8,9,10])
    ax.set_xticklabels([-180,-90,0,90,180])
    
    keys = fad.get_keys_with_attr(dataset, ['odor_stimulus'], ['on'])
    ax = fig.add_subplot(133)
    plot_time_since_odor_and_heading_on_ax(ax, config, dataset, keys, auto_velocity_adjustment=True)
    ax.vlines(-np.pi/2., 0, 10, color='black')
    ax.vlines(np.pi/2., 0, 10, color='black')
    ax.set_title('relative to visual motion')
    fpl.adjust_spines(ax, ['bottom'], xticks=[-np.pi, -np.pi/2., 0, np.pi/2., np.pi], yticks=[0,1,2,3,4,5,6,7,8,9,10])
    ax.set_xticklabels([-180,-90,0,90,180])
    ax.set_xlabel('heading\nrelative to ground motion')
    
    
    path = config.path
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'heading_vs_time_since_odor.pdf')
    
    print 'SAVING TO: ', pdf_name_with_path
    
    fig.savefig(pdf_name_with_path, format='pdf')
    
    
################################################################################################3

def plot_time_since_odor_and_altitude_on_ax(ax, config, dataset, keys, keys_to_show=[]):

    if keys is None:
        keys = dataset.trajecs.keys()
        
    altitudes = []
    time_since_odor = []
    
    
        
    for key in keys:
        trajec = dataset.trajecs[key]
        frames_where_odor = np.where(trajec.odor>10)[0]
        
        headings_for_trajec = []
        time_since_odor_for_trajec = []
        
        for f in range(trajec.length):
            tmp = f - np.array(frames_where_odor)
            try:
                tso = tmp[np.where(tmp>0)[0][-1]]/trajec.fps
            except:
                tso = 10000
                
            if tso < 20 and tso > 0.01:
                altitudes.append(trajec.positions[f,2])
                time_since_odor.append(tso)
                
            
    binsx = np.linspace(-.15, .15, 100)
    binsy = np.linspace(0,10,100)
    fpl.histogram2d(ax, altitudes, time_since_odor, bins=(binsx, binsy), logcolorscale=True)            
    
    ax.set_aspect('auto')
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(0,10)
    

def plot_time_since_odor_and_altitude(config, dataset):
    
    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    keys_noodor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'none')
    ax = fig.add_subplot(121)
    plot_time_since_odor_and_altitude_on_ax(ax, config, dataset, keys_noodor)
    ax.set_title('odor off')
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=[-.15, 0, .15], yticks=[0,1,2,3,4,5,6,7,8,9,10])
    ax.set_xlabel('altitude')
    ax.set_ylabel('time since odor, sec')

    keys_odor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'on')
    ax = fig.add_subplot(122)
    keys_to_show = ['20120926_174720_6196']
    plot_time_since_odor_and_altitude_on_ax(ax, config, dataset, keys_odor)
    ax.set_title('odor on')
    fpl.adjust_spines(ax, ['bottom'], xticks=[-.15,0,.15], yticks=[0,1,2,3,4,5,6,7,8,9,10])
    ax.set_xlabel('heading')
    
    path = config.path
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'altitude_vs_time_since_odor.pdf')
    
    print 'SAVING TO: ', pdf_name_with_path
    
    fig.savefig(pdf_name_with_path, format='pdf')
    
    
def plot_angular_velocity_vs_speed(config, dataset, keys=None):
    
    angular_vel = []
    speed = []
    
    if keys is None:
        keys = dataset.trajecs.keys()
    
    for key in keys:
        trajec = dataset.trajecs[key]
        speed.extend(trajec.speed)
        angular_vel.extend(trajec.heading_smooth_diff)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    binsx = np.linspace(0, 20, 100)
    binsy = np.linspace(0, 1,100)
    fpl.histogram2d(ax, np.abs(np.array(angular_vel)), np.array(speed), bins=(binsx,binsy), logcolorscale=True)
    
    ax.set_xlim(0,20)
    ax.set_ylim(0,1)
    ax.set_aspect('auto')
    
    path = config.path
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'angular_velocity_vs_speed.pdf')
    
    print 'SAVING TO: ', pdf_name_with_path
    
    fig.savefig(pdf_name_with_path, format='pdf')
    
    
    
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
    
    keys = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'on')

    plot_heading_time_to_odor_relative(config, dataset, attribute='heading_smooth', column=0, relative_to='exit', threshold_odor=10)

    
    
    
    
        
        
        
        
