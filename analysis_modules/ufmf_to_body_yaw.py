import sys, os
from optparse import OptionParser
import pickle
import imp

import fly_plot_lib
fly_plot_lib.set_params.pdf()
import fly_plot_lib.plot as fpl
import matplotlib.pyplot as plt

import flydra_analysis_tools.flydra_analysis_dataset as fad
from flydra_analysis_tools import floris_math
from flydra_analysis_tools import kalman_math

from flydra_analysis_tools import numpyimgproc as nim
import motmot.ufmf.ufmf as ufmf

import copy
import numpy as np

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
    heading_smooth_diff = xsmooth[:,1]*100.
    
    heading = floris_math.fix_angular_rollover(heading_norollover)
    heading_smooth = floris_math.fix_angular_rollover(heading_norollover_smooth)
    
    flipm = np.where(heading_smooth < 0)[0]
    flipp = np.where(heading_smooth > 0)[0]
    
    heading_smooth[flipm] += np.pi
    heading_smooth[flipp] -= np.pi
    
    return heading_smooth

def get_heading_and_orientation(dataset, orientation_datafile, keys=None):
    datafile = open(orientation_datafile, 'r')
    data = pickle.load(datafile)
    timestamp = []
    orientation = []
    eccentricity = []
    for frame in data.keys():
        framedata = data[frame]
        if framedata is not None:
            timestamp.append(framedata['timestamp'])
            orientation.append(framedata['orientation'])
            eccentricity.append(framedata['eccentricity'])
    timestamp_ufmf = np.array(timestamp)
    orientation_ufmf = np.array(orientation)
    eccentricity_ufmf = np.array(eccentricity)
    
    heading_for_hist = []
    orientation_for_hist = []
    eccentricity_for_hist = []
    speed_for_hist = []
    velocities = None
    
    if keys is None:
        keys = dataset.trajecs.keys()
    for key in keys:
        trajec = dataset.trajecs[key]
        saccade_frames = saccade_frames = [item for sublist in trajec.saccades for item in sublist]
        timestamps_fly = trajec.timestamp_epoch + trajec.time_fly
        tdiff = np.abs(timestamp_ufmf-timestamps_fly[0])
        if np.min(tdiff) > 100:
            continue
        
        for f, t in enumerate(timestamps_fly):
            if f in saccade_frames:
                continue
            if np.abs(trajec.positions[f,2]) > 0.1:
                continue
            if trajec.positions[f,0] < -0.1:
                continue
            if  trajec.positions[f,0] > 0.65:
                continue
            if np.abs(trajec.positions[f,1]) > 0.1:
                continue
            if np.abs(trajec.velocities[f,2]) > 0.05:
                continue
            

            tdiff = np.abs(timestamp_ufmf-t)
            if np.min(tdiff) < 0.009:
                index = np.argmin(tdiff)
                print eccentricity_ufmf[index]
                if eccentricity_ufmf[index] < 1:
                    orientation_for_hist.append(orientation_ufmf[index])
                    eccentricity_for_hist.append(eccentricity_ufmf[index])
                    speed_for_hist.append(trajec.speed[f])
                    
                    if velocities is None:
                        velocities = copy.copy(trajec.velocities[f])
                    else:
                        velocities = np.vstack((velocities, trajec.velocities[f]))
                    
                    heading = trajec.heading_smooth[f]
                    # flip
                    if heading < 0:
                        heading_for_hist.append(heading + np.pi)
                    else:
                        heading_for_hist.append(heading - np.pi)
    
    return np.array(heading_for_hist), -1*np.array(orientation_for_hist), np.array(eccentricity_for_hist), np.array(speed_for_hist), velocities
    

def plot_eccentricity_vs_orientation(path, orientation_datafile, savename='orientation_vs_eccentricity.pdf'):
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    
    timestamps,x,y,orientation,eccentricity = load_data(orientation_datafile, return_zeros_for_no_data=False)
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    
    ax.plot(orientation, eccentricity, '.', markersize=1)
    
    xticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    xticklabels = ['-180', '-90', 'upwind', '90', '180']
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('orientation')
    ax.set_ylabel('eccentricity')

    path = config.path
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    fig_name_with_path = os.path.join(save_figure_path, savename)

    print 'SAVING TO: ', fig_name_with_path
    fig.savefig(fig_name_with_path, format='pdf')
    
def plot_eccentricity_vs_speed_xy(path, orientation_datafile, savename='eccentricity_vs_speed_xy.pdf'):
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    
    culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
    dataset = fad.load(culled_dataset_filename)
    #keys = fad.get_keys_with_attr(dataset, ['odor_stimulus'], ['none'])
    keys = dataset.trajecs.keys()

    heading_for_hist, orientation_for_hist, eccentricity_for_hist, speed_for_hist, velocities = get_heading_and_orientation(dataset, orientation_datafile, keys=keys)
    
    # airspeed headings
    velocities_air = copy.copy(velocities)
    velocities_air[:,0] -= 0.4
    heading_air = calc_heading(velocities_air)
    
    airspeed = np.array([np.linalg.norm(velocities_air[i,0:2]) for i in range(velocities_air.shape[0])])
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    
    xlim = [-1,1]
    ylim = [0,1]
    fpl.scatter(ax, velocities[:,0], eccentricity_for_hist, color=heading_air, colornorm=[-.4,.4], radius=.005, xlim=xlim, ylim=ylim)
    
    fpl.adjust_spines(ax, ['left', 'bottom'])
    
    ax.set_xlabel('airspeed')
    ax.set_ylabel('eccentricity (tight correlation with pitch)')

    path = config.path
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    fig_name_with_path = os.path.join(save_figure_path, savename)

    print 'SAVING TO: ', fig_name_with_path
    fig.savefig(fig_name_with_path, format='pdf')
    

def plot_eccentricity_vs_heading(path, orientation_datafile, savename='heading_vs_eccentricity_xy.pdf'):
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    
    culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
    dataset = fad.load(culled_dataset_filename)
    #keys = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['on', 'downwind'])
    keys = dataset.trajecs.keys()

    heading_for_hist, orientation_for_hist, eccentricity_for_hist, speed_for_hist, velocities = get_heading_and_orientation(dataset, orientation_datafile, keys=keys)
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    
    xlim = [0, np.pi]
    ylim = [0,1]
    fpl.scatter(ax, np.abs(heading_for_hist), np.array(eccentricity_for_hist), color=np.array(speed_for_hist), colornorm=[0.1,0.5], radius=.0035, xlim=xlim, ylim=ylim)
    #ax.plot(heading_for_hist, eccentricity_for_hist, '.', markersize=1)
    
    xticks = [0, np.pi/2., np.pi]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    xticklabels = ['upwind', '90', '180']
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('groundspeed heading')
    ax.set_ylabel('eccentricity')
    

    path = config.path
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    fig_name_with_path = os.path.join(save_figure_path, savename)

    print 'SAVING TO: ', fig_name_with_path
    fig.savefig(fig_name_with_path, format='pdf')
    

def plot_orientation_vs_heading(path, orientation_datafile, savename='heading_vs_orientation_xy.pdf'):
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    
    culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
    dataset = fad.load(culled_dataset_filename)
    #keys = fad.get_keys_with_attr(dataset, ['odor_stimulus', 'visual_stimulus'], ['on', 'none'])
    keys = dataset.trajecs.keys()

    heading_for_hist, orientation_for_hist, eccentricity_for_hist, speed_for_hist, velocities = get_heading_and_orientation(dataset, orientation_datafile, keys=keys)
    
    # airspeed headings
    velocities_air = copy.copy(velocities)
    velocities_air[:,0] -= 0.4
    heading_air = calc_heading(velocities_air)
    
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    
    xlim = [-np.pi, np.pi]
    ylim = [-np.pi, np.pi]
    
    heading_type = 'groundspeed'
    
    if heading_type == 'airspeed':
        x = np.array(heading_air)
    elif heading_type == 'groundspeed':
        x = np.array(heading_for_hist)
    fpl.scatter(ax, x, np.array(orientation_for_hist), color=speed_for_hist, colornorm=[0.1,0.6], radius=.01, xlim=xlim, ylim=ylim)
    #ax.plot(heading_for_hist, eccentricity_for_hist, '.', markersize=1)
    
    xticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=xticks)
    xticklabels = ['-180', '-90', 'upwind', '90', '180']
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(xticklabels)
    
    if heading_type == 'airspeed':
        ax.set_xlabel('airspeed heading')
    elif heading_type == 'groundspeed':
        ax.set_xlabel('groundspeed heading')
    
    ax.set_ylabel('body orientation')
    

    path = config.path
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    fig_name_with_path = os.path.join(save_figure_path, savename)

    print 'SAVING TO: ', fig_name_with_path
    fig.savefig(fig_name_with_path, format='pdf')
    
    
def plot_orientation_histogram(path, orientation_datafile, keys=None, savename='orientation_histogram.pdf'):
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    
    if 1:
        culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
        dataset = fad.load(culled_dataset_filename)
        keys = fad.get_keys_with_attr(dataset, ['odor_stimulus'], ['on'])
    
        heading_for_hist, orientation_for_hist, eccentricity_for_hist, speed_for_hist, velocities = get_heading_and_orientation(dataset, orientation_datafile, keys=keys)
    
    else:
        timestamps,x,y,orientation_for_hist,eccentricity = load_data(orientation_datafile, return_zeros_for_no_data=False)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)

    bins = 100
    
    orientation_for_hist = np.array(orientation_for_hist)
    orientation_for_hist = floris_math.fix_angular_rollover(orientation_for_hist)
    
    heading_for_hist = np.array(heading_for_hist)
    
    fpl.histogram(ax, [orientation_for_hist, heading_for_hist], bins=bins, bin_width_ratio=1, colors=['black', 'red'], edgecolor='none', bar_alpha=1, curve_fill_alpha=0.2, curve_line_alpha=1, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False)

    xticks = [-np.pi, -np.pi/2., 0, np.pi/2., np.pi]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    xticklabels = ['-180', '-90', 'upwind', '90', '180']
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('orientation')
    ax.set_ylabel('occurences, normalized')

    path = config.path
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path=os.path.join(figure_path, 'odor_traces/')
        
    figure_path = os.path.join(path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    fig_name_with_path = os.path.join(save_figure_path, savename)

    print 'SAVING TO: ', fig_name_with_path
    fig.savefig(fig_name_with_path, format='pdf')




def load_data(datafile, return_zeros_for_no_data=False):
    # unpack data from the pickled file - this will vary depending on your data, of course
    datafile = open(datafile, 'r')
    data = pickle.load(datafile)
    timestamps = []
    x = []
    y = []
    orientation = []
    eccentricity = []
    for frame in data.keys():
        framedata = data[frame]
        if framedata is not None:
            timestamps.append(framedata['timestamp'])
            x.append(framedata['position'][0])
            y.append(framedata['position'][1])
            orientation.append(framedata['orientation'])
            eccentricity.append(framedata['eccentricity'])
        elif return_zeros_for_no_data:
            timestamps.append(0)
            x.append(0)
            y.append(0)
            orientation.append(0)
            eccentricity.append(0)
    
    return timestamps,x,y,orientation,eccentricity


def extract_unsigned_orientation_and_position(img):
    center, longaxis, shortaxis, body, ratio = nim.find_ellipse(img, background=None, threshrange=[0,120], sizerange=[10,350], erode=False)
    unsigned_orientation = np.arctan2(longaxis[0], longaxis[1])
    position = center[::-1]
    if ratio[0] is not None:
        eccentricity = ratio[1] / ratio[0]
    else:
        eccentricity = None
    return position, unsigned_orientation, eccentricity         
    
def extract_signed_orientation(img, prev_pos=None):
    position, unsigned_orientation, eccentricity = extract_unsigned_orientation_and_position(img)
    if prev_pos is not None:
        velocity = position - prev_pos
    else:
        velocity = np.zeros_like(position)
        
    # find signed orientation...
    velocity_heading = np.arctan2(velocity[1], velocity[0])
    if np.dot(unsigned_orientation, velocity_heading) < 0:
        orientation = unsigned_orientation + np.pi
    else:
        orientation = unsigned_orientation
        
    # unwrap orientation:
    orientation = floris_math.fix_angular_rollover(orientation)
    
    # flip it
    if orientation < 0:
        orientation = orientation + np.pi
    else:
        orientation = orientation - np.pi
        
    return position, orientation, eccentricity
    
    


def main(filename, start, end, saveimages=None):#'/home/caveman/DATA/tmp_orientation_checks/images'):
    orientation_frames = {}
    movie = ufmf.FlyMovieEmulator(filename)
    if end == -1:
        end = movie.get_n_frames()
    
    prev_pos = None
    for frame in range(start, end):
        if (frame/100.) == int(frame/100.):
            print frame
        img = -1*(movie.get_mean_for_timestamp(movie.get_frame(frame)[1]) - movie.get_frame(frame)[0])
        
        if saveimages is not None:
            fstr = str(frame)+'.png'
            imname = os.path.join(saveimages, fstr)
            plt.imsave(imname, img)
        
        if np.min(img) < -50:
            timestamp = movie.get_frame(frame)[1]
            prev_pos, orientation, eccentricity = extract_signed_orientation(img, prev_pos=prev_pos)
            framedata = {'frame': frame, 'timestamp': timestamp, 'orientation': orientation, 'eccentricity': eccentricity, 'position': prev_pos}
            orientation_frames.setdefault(frame, framedata)
        else:
            framedata = None
            orientation_frames.setdefault(frame, framedata)
        
    return orientation_frames










def process_ufmf(options):
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(options.path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(options.path)
    orientation_frames = main(options.file, options.start, options.stop)
    
    filename = 'DATA_' + os.path.basename(options.file).split('.')[0]
    ufmf_path = os.path.join(options.path, 'data', 'ufmfs')   
    filename_with_path = os.path.join(ufmf_path, filename) 
        
    f = open(filename_with_path, 'w')
    pickle.dump(orientation_frames, f)
    f.close()






if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--file", type="str", dest="file", default='',
                        help="path to ufmf you wish to process")
    parser.add_option("--start", type=int, dest="start", default=0,
                        help="first frame")
    parser.add_option("--stop", type=int, dest="stop", default=-1,
                        help="last frame")
    parser.add_option("--action", type="str", dest="action", default='process_ufmf',
                        help="what do you want to do?") 
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to dataset config file") 
    parser.add_option("--orientation", type="str", dest="orientation", default='',
                        help="path to orientation data") 
                        
    (options, args) = parser.parse_args()
    
    if options.action == 'process_ufmf':
        process_ufmf(options)
    elif options.action == 'histogram_plot':
        plot_orientation_histogram(options.path, options.orientation)
    elif options.action == 'orientation_vs_eccentricity':
        plot_eccentricity_vs_orientation(options.path, options.orientation)
    elif options.action == 'eccentricity_vs_speed':
        plot_eccentricity_vs_speed_xy(options.path, options.orientation)
    elif options.action == 'eccentricity_vs_heading':
        plot_eccentricity_vs_heading(options.path, options.orientation)
    elif options.action == 'orientation_vs_heading':
        plot_orientation_vs_heading(options.path, options.orientation)
        
        
        
        
    
    
    
