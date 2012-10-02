#!/usr/bin/env python
import sys, os
sys.path.append('../')
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




def plot_deceleration(config, dataset, save=True):
    
    keys = fad.get_keys_with_attr(dataset, 'post_behavior', 'landing')

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    
    speeds = []
    angles = []

    for key in keys:
        trajec = dataset.trajecs[key]
        
        # get landing frame
        for i, behavior in enumerate(trajec.post_behavior):
            if behavior == 'landing':
                landing_frame = trajec.post_behavior_frames[i]
                break
        
        if landing_frame > 60:
            frame0 = landing_frame-60
            frame1 = np.min([landing_frame+5, trajec.length])
            frames = np.arange(frame0, frame1).tolist()
            
            angle_subtended_by_post = 2*np.sin(config.post_radius/(trajec.distance_to_post+config.post_radius))

            # find frame of deceleration
            # step backwards from frame of landing until acceleration is positive
            accel = -1
            f = landing_frame
            while accel < 0:
                f -= 1
                accel = trajec.speed_xy[f] - trajec.speed_xy[f-1]
                
            speeds.append(trajec.speed[f])
            angles.append(np.log(angle_subtended_by_post[f]))
            
            ax.plot(np.log(angle_subtended_by_post[frames]), trajec.speed[frames], color='black')
            ax.plot(np.log(angle_subtended_by_post[f]), trajec.speed[f], '.', color='purple', markersize=8, zorder=10)
        
    speeds = np.array(speeds)
    angles = np.array(angles)
        
    lm = data_fit.models.LinearModel()
    lm.fit(speeds, inputs=angles)

    xvals = np.linspace(np.log(5*np.pi/180.), np.log(np.pi/2.), 10)
    yvals = lm.get_val(xvals)
    
    ax.plot(xvals, yvals, color='purple', linewidth=2)

    angle_ticks = [5, 10, 30, 60, 90, 180]
    xticks = np.log(np.array(angle_ticks)*np.pi/180.)
    yticks = [0, .2, .4, .6, .8]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks)
    ax.set_xticklabels(angle_ticks)

    ax.set_xlabel('Retinal size of post')
    ax.set_ylabel('Ground speed')
    
    if save:
        figure_path = os.path.join(config.path, config.figure_path)
        save_figure_path=os.path.join(figure_path, 'activity/')
        figname = save_figure_path + 'deceleration_for_landings' + '.pdf'
        plt.savefig(figname, format='pdf')        
        
        
if __name__ == '__main__':
    plot_deceleration(config, dataset)

