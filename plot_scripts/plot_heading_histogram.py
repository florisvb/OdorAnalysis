#!/usr/bin/env python
import sys, os
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

headings = []

keys_odor = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['on', 'none'])
keys_noodor = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['none', 'none'])

threshold_odor = 10

# odor, experienced
print 'odor, experienced'
keys = keys_odor
headings = []
for key in keys:
    trajec = dataset.trajecs[key]
    if np.max(trajec.odor) < threshold_odor:
        continue
    frames_where_odor = np.where(trajec.odor > threshold_odor)[0]
    # find non saccade segments:
    saccade_frames = [item for sublist in trajec.saccades for item in sublist]
    
    for f, h in enumerate(trajec.heading_smooth):
        if f not in saccade_frames:
            if np.abs(trajec.positions[f,2]) > 0.08: # skip low and high points
                continue
        
            tmp = f - np.array(frames_where_odor)
            try:
                time_since_odor = tmp[np.where(tmp>0)[0][-1]]/trajec.fps
            except:
                time_since_odor = 10000
            if time_since_odor > 0.5 and time_since_odor < 3:
                headings.append(h)
headings = np.array(headings)
headings_flipped = []
for h in headings:
    if h < 0:
        headings_flipped.append(h+np.pi)
    else:
        headings_flipped.append(h-np.pi)
headings_flipped_odor = np.array(headings_flipped)

print 'odor experienced recently'
# odor, experienced recently
keys = keys_odor
headings = []
for key in keys:
    trajec = dataset.trajecs[key]
    if np.max(trajec.odor) < threshold_odor:
        continue
    frames_where_odor = np.where(trajec.odor > threshold_odor)[0]
    # find non saccade segments:
    saccade_frames = [item for sublist in trajec.saccades for item in sublist]
    
    for f, h in enumerate(trajec.heading_smooth):
        if f not in saccade_frames:
            if np.abs(trajec.positions[f,2]) > 0.08: # skip low and high points
                continue
                
            tmp = f - np.array(frames_where_odor)
            try:
                time_since_odor = tmp[np.where(tmp>0)[0][-1]]/trajec.fps
            except:
                time_since_odor = 10000
            if time_since_odor < 0.5:
                headings.append(h)
headings = np.array(headings)
headings_flipped = []
for h in headings:
    if h < 0:
        headings_flipped.append(h+np.pi)
    else:
        headings_flipped.append(h-np.pi)
headings_flipped_odor_recent = np.array(headings_flipped)


print 'odor not experienced'
# odor, not experienced
keys = keys_odor
headings = []
for key in keys:
    trajec = dataset.trajecs[key]
    if np.max(trajec.odor) > threshold_odor:
        continue
    # find non saccade segments:
    saccade_frames = [item for sublist in trajec.saccades for item in sublist]
    
    for f, h in enumerate(trajec.heading_smooth):
        if f not in saccade_frames:
            if np.abs(trajec.positions[f,2]) > 0.08: # skip low and high points
                continue
            headings.append(h)
    
headings = np.array(headings)
headings_flipped = []
for h in headings:
    if h < 0:
        headings_flipped.append(h+np.pi)
    else:
        headings_flipped.append(h-np.pi)
headings_flipped_odor_false = np.array(headings_flipped)

print 'no odor, experienced'
# no odor, experienced
keys = keys_noodor
headings = []
for key in keys:
    trajec = dataset.trajecs[key]
    if np.max(trajec.odor) < threshold_odor:
        continue
    
    saccade_frames = [item for sublist in trajec.saccades for item in sublist]
    for f, h in enumerate(trajec.heading_smooth):
        if f not in saccade_frames:
            if np.abs(trajec.positions[f,2]) > 0.08: # skip low and high points
                continue
            headings.extend(trajec.heading_smooth.tolist())
headings = np.array(headings)
headings_flipped = []
for h in headings:
    if h < 0:
        headings_flipped.append(h+np.pi)
    else:
        headings_flipped.append(h-np.pi)
headings_flipped_noodor = np.array(headings_flipped)

print 'no odor, not experienced'
# no odor, not experienced
keys = keys_noodor
headings = []
for key in keys:
    trajec = dataset.trajecs[key]
    if np.max(trajec.odor) > threshold_odor:
        continue
    saccade_frames = [item for sublist in trajec.saccades for item in sublist]
    for f, h in enumerate(trajec.heading_smooth):
        if f not in saccade_frames:
            if np.abs(trajec.positions[f,2]) > 0.08: # skip low and high points
                continue
            headings.extend(trajec.heading_smooth.tolist())
headings = np.array(headings)
headings_flipped = []
for h in headings:
    if h < 0:
        headings_flipped.append(h+np.pi)
    else:
        headings_flipped.append(h-np.pi)
headings_flipped_noodor_false = np.array(headings_flipped)

data = [headings_flipped_odor_recent, headings_flipped_odor, headings_flipped_odor_false, headings_flipped_noodor, headings_flipped_noodor_false]

    
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111)

bins = 100

print 'making histogram'
fpl.histogram(ax, data, bins=bins, bin_width_ratio=1, colors=['red', 'green', 'purple', 'black', 'blue'], edgecolor='none', bar_alpha=1, curve_fill_alpha=0.2, curve_line_alpha=1, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False)

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
fig_name_with_path = os.path.join(save_figure_path, 'heading_histogram.pdf')

ax.text(np.pi/2., 0.7, 'time since odor: $<0.5$ sec', color='red')
ax.text(np.pi/2., 0.6, 'time since odor: $>0.5, <3$ sec', color='green')
ax.text(np.pi/2., 0.5, 'odor on, not passed through plume', color='purple')
ax.text(np.pi/2., 0.4, 'no odor, passed through plume', color='black')
ax.text(np.pi/2., 0.3, 'no odor, not passed through plume', color='blue')


fig.savefig(fig_name_with_path, format='pdf')




