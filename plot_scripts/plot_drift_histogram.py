#!/usr/bin/env python
import sys, os
import copy
sys.path.append('../')
sys.path.append('../analysis_modules')
import flydra_analysis_tools as fat
import fly_plot_lib
fly_plot_lib.set_params.pdf()
import fly_plot_lib.plot as fpl
fad = fat.flydra_analysis_dataset
dac = fat.dataset_analysis_core
fap = fat.flydra_analysis_plot
tac = fat.trajectory_analysis_core

from flydra_analysis_tools import floris_math
from flydra_analysis_tools import kalman_math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

heading = []

keys_odor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'on')
keys_noodor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'none')

threshold_odor = 10

def calc_heading(velocities):
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
    heading_smooth_diff = xsmooth[:,1]*trajec.fps
    
    heading = floris_math.fix_angular_rollover(heading_norollover)
    heading_smooth = floris_math.fix_angular_rollover(heading_norollover_smooth)
    
    return heading_smooth
    
    

# odor, experienced
keys = keys_odor
velocities = None
for key in keys:
    trajec = dataset.trajecs[key]
    if np.max(trajec.odor) < threshold_odor:
        continue
    frames_where_odor = np.where(trajec.odor > threshold_odor)[0]
    # find non saccade segments:
    saccade_frames = [item for sublist in trajec.saccades for item in sublist]
    
    for f, h in enumerate(trajec.heading_smooth):
        if f not in saccade_frames:
        
            tmp = f - np.array(frames_where_odor)
            try:
                time_since_odor = tmp[np.where(tmp>0)[0][-1]]/trajec.fps
            except:
                time_since_odor = 10000
            if time_since_odor > 0 and time_since_odor < 1000:
            
                if velocities is None:
                    velocities = copy.copy(trajec.velocities[f,:])
                else:
                    velocities = np.vstack((velocities,trajec.velocities[f,:]))
            

heading_smooth_real = calc_heading(velocities)


# simulate drift.. maintain same airspeed
velocities_drift = copy.copy(velocities)
velocities_drift[:,0] += 0.1
heading_smooth_drift_downwind = calc_heading(velocities_drift)

# simulate drift.. maintain same airspeed
velocities_drift = copy.copy(velocities)
velocities_drift[:,0] -= 0.1
heading_smooth_drift_upwind = calc_heading(velocities_drift)


heading_flipped_drift_downwind = []
for h in heading_smooth_drift_downwind:
    if h < 0:
        heading_flipped_drift_downwind.append(h+np.pi)
    else:
        heading_flipped_drift_downwind.append(h-np.pi)
heading_flipped_drift_downwind = np.array(heading_flipped_drift_downwind)

heading_flipped_drift_upwind = []
for h in heading_smooth_drift_upwind:
    if h < 0:
        heading_flipped_drift_upwind.append(h+np.pi)
    else:
        heading_flipped_drift_upwind.append(h-np.pi)
heading_flipped_drift_upwind = np.array(heading_flipped_drift_upwind)

heading_flipped_real = []
for h in heading_smooth_real:
    if h < 0:
        heading_flipped_real.append(h+np.pi)
    else:
        heading_flipped_real.append(h-np.pi)
heading_flipped_real = np.array(heading_flipped_real)


data = [heading_flipped_real, heading_flipped_drift_downwind, heading_flipped_drift_upwind]

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)

bins = 100

fpl.histogram(ax, data, bins=bins, bin_width_ratio=1, colors=['black', 'green', 'red'], edgecolor='none', bar_alpha=1, curve_fill_alpha=0.2, curve_line_alpha=1, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False)

xticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
xticklabels = ['-180', '-90', 'upwind', '90', '180']
ax.set_xticklabels(xticklabels)
ax.set_xlabel('heading')
ax.set_ylabel('occurences, normalized')

path = config.path
figure_path = os.path.join(config.path, config.figure_path)
save_figure_path=os.path.join(figure_path, 'odor_traces/')
    
figure_path = os.path.join(path, config.figure_path)
save_figure_path = os.path.join(figure_path, 'odor_traces/')
fig_name_with_path = os.path.join(save_figure_path, 'heading_histogram_drifting.pdf')

#ax.text(np.pi/4., 0.6, 'if there were no wind, but flies flew\nat the same airspeed,\ntheir ground velocity would be\n$= actual ground vel - wind vel$', color='red', verticalalignment='top')
#ax.text(np.pi/4., 0.7, 'actual heading', color='black')


fig.savefig(fig_name_with_path, format='pdf')


