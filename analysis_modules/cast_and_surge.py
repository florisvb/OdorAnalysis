import numpy as np
import fly_plot_lib.plot as fpl
from fly_plot_lib import flymath

import matplotlib.pyplot as plt
import os

import matplotlib.patches as patches

import flydra_analysis_tools.trajectory_analysis_core as tac

def get_frames_where_casting_and_surging(trajec):

    threshold_odor = 10
    
    cast_frames = []
    surge_frames = []
    for f in range(trajec.length):
        if np.abs(trajec.heading_smooth[f]) < 15*np.pi/180.:
            frame0 = np.max([0,f-50])
            
            try:
                if np.max(trajec.odor[frame0:f]) > threshold_odor:
                    surge_frames.append(f)
            except:
                pass
                
            surge_frames.append(f)
            
        elif np.abs(trajec.heading_smooth[f]) > 80*np.pi/180. and np.abs(trajec.heading_smooth[f]) < 100*np.pi/180:
            frame0 = np.max([0,f-200])
            frame1 = np.max([0,f-50])
            
            try:
                if np.max(trajec.odor[frame0:f]) > threshold_odor:
                    if np.max(trajec.odor[frame1:f]) < threshold_odor:
                        cast_frames.append(f)
            except:
                pass
                
    if len(cast_frames) > 10:
        cast_frame_chunks, break_points = flymath.get_continuous_chunks(cast_frames, jump=2)
        cast_frames = []
        for chunk in cast_frame_chunks:
            if len(chunk) > 15:
                cast_frames.extend(chunk)
    else:
        cast_frames = []
            
    if len(surge_frames) > 10:
        surge_frame_chunks, break_points = flymath.get_continuous_chunks(surge_frames, jump=2)
        surge_frames = []
        for chunk in surge_frame_chunks:
            if len(chunk) > 15:
                surge_frames.extend(chunk)
    else:
        surge_frames = []
        
    return cast_frames, surge_frames
    
    
def get_mean_speed_in_cast(dataset, keys=None):
    if keys is None:
        keys = dataset.trajecs.keys()
        
    cast_speeds = []
    surge_speeds = []
    for key in keys:
        print key
        trajec = dataset.trajecs[key]
        cast_frames, surge_frames = get_frames_where_casting_and_surging(trajec)
        cast_speeds.extend((trajec.speed_xy[cast_frames]).tolist())
        surge_speeds.extend((trajec.speed_xy[surge_frames]).tolist())
    
    return cast_speeds, surge_speeds
    
def plot_cast_surge(cast_speeds, surge_speeds):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.histogram(ax, [np.array(cast_speeds),np.array(surge_speeds)], bins=50, colors=['red','green'], normed=False)
    
    
    #print 'mean cast speeds: ', np.mean(cast_speeds), ' +/- ', np.std(cast_speeds)
    #print 'mean surge speeds: ', np.mean(surge_speeds), ' +/- ', np.std(surge_speeds)
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
