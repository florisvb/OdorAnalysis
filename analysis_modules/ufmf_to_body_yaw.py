import sys, os
from optparse import OptionParser
import pickle
import imp

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import patches

import fly_plot_lib
fly_plot_lib.set_params.pdf()
import fly_plot_lib.plot as fpl
import fly_plot_lib.flymath as flymath
import fly_plot_lib.animate as flyanim
import matplotlib.pyplot as plt

import flydra_analysis_tools.flydra_analysis_dataset as fad
from flydra_analysis_tools import floris_math
from flydra_analysis_tools import kalman_math

from flydra_analysis_tools import numpyimgproc as nim
import motmot.ufmf.ufmf as ufmf

import copy
import numpy as np

import time

##########################################################################################################
# load and interpret data

def get_keys_for_ufmf_frames(ufmf_frames, ufmf_frame_offset, frame_to_key):
    
    keys = []
    keys_and_ufmf_frames = {}
    for ufmf_frame in ufmf_frames:
        flydra_frame = ufmf_frame + ufmf_frame_offset
        if frame_to_key.has_key(flydra_frame):
            keys_for_frame = frame_to_key[flydra_frame]
            for key in keys_for_frame:
                if key not in keys:
                    keys.append(key)
                    keys_and_ufmf_frames.setdefault(key, [ufmf_frame])
                else:
                    keys_and_ufmf_frames[key].append(ufmf_frame)
    return keys, keys_and_ufmf_frames

def get_ufmf_frame_offset(dataset, ufmf_frames, ufmf_timestamps, npts=25):
    # find frames where difference < 0.008
    frame_offsets = []
    
    n = 0
    smallest_tdiff = 100000
    for key in dataset.trajecs.keys():
        print key
        trajec = dataset.trajecs[key]
        if np.min(np.abs(trajec.timestamp_epoch - np.array(ufmf_timestamps))) > 200:
            continue
        for f, t in enumerate(trajec.timestamp_epoch + trajec.time_fly):
            tdiff = np.abs(t - np.array(ufmf_timestamps))
            if np.min(tdiff) < 0.008:
                n += 1
                print np.min(tdiff)
                if n > npts:
                    return int(np.mean(np.array(frame_offsets)))
                    #return frame_offsets
                ufmf_timestamps_index = np.argmin(tdiff)
                ufmf_frame = ufmf_frames[ufmf_timestamps_index]
                frame_offset = trajec.first_frame + f - ufmf_frame
                frame_offsets.append(frame_offset)
                #break # only one data point per fly
    return int(np.mean(np.array(frame_offsets)))-1
    

def in_range(val, minmax):
    if val > minmax[0] and val < minmax[1]:
        return True
    else:
        return False

def is_position_in_volume(pos, x_range, y_range, z_range):
    inx = in_range(pos[0], x_range)
    iny = in_range(pos[1], y_range)
    inz = in_range(pos[2], z_range)
    return inx*iny*inz
    
def get_keys_in_volume_for_flydra_frame(flydra_frame, frame_to_key, dataset):
            
    keys = frame_to_key[flydra_frame]
    keys_in_volume = []
    for key in keys:
        trajec = dataset.trajecs[key]
        flydra_frames = (trajec.first_frame + np.arange(0, trajec.length)).tolist()
        index = flydra_frames.index(flydra_frame)
        pos = trajec.positions[index]
        in_volume = is_position_in_volume(pos, [-.2,.4], [-.18,.18], [-.18,.18])
        if in_volume:
            keys_in_volume.append(key)
    return keys_in_volume
    
def get_camera_frames_with_keys_in_volume(frame_to_key, dataset):
    camera_frames = []
    for camera_frame in np.sort(frame_to_key.keys()):
        if int(camera_frame/100.) == camera_frame/100.:
            print camera_frame
        keys_in_volume = get_keys_in_volume_for_camera_frame(camera_frame, frame_to_key, dataset)
        if len(keys_in_volume) > 0:
            camera_frames.append(camera_frame)
            
    # find trajec with ufmf_frame_offset
    
    ufmf_frame_offset = find_trajec_with_ufmf_frame_offset(dataset)
    ufmf_frames = np.array(camera_frames) - ufmf_frame_offset
        
    return ufmf_frames
    
    
def find_trajec_with_ufmf_frame_offset(dataset):
    ufmf_frame_offset = None
    n = 0
    while ufmf_frame_offset is None:
        try:
            ufmf_frame_offset = dataset.get_trajec(n).ufmf_frame_offset
        except:
            n += 1
    return ufmf_frame_offset
    
def load_ufmf_data_from_dict(orientation_datafile):
    datafile = open(orientation_datafile, 'r')
    data = pickle.load(datafile)
    
    timestamp = []
    longaxis = []
    eccentricity = []
    ufmf_frames = []
    position = []
    for frame in data.keys():
        framedata = data[frame]
        if framedata is not None:
            timestamp.append(framedata['timestamp'])
            longaxis.append(framedata['longaxis'])
            eccentricity.append(framedata['eccentricity'])
            ufmf_frames.append(frame)
            position.append(framedata['position'])
            
    timestamp_ufmf = timestamp
    #longaxis_ufmf = np.array(longaxis)
    eccentricity_ufmf = eccentricity
    ufmf_frames_ufmf = ufmf_frames
    return timestamp_ufmf, longaxis, eccentricity_ufmf, ufmf_frames_ufmf, position
    
    
def load_ufmf_data_from_dict_list(orientation_datafile_list):
    ufmf_data = load_ufmf_data_from_dict(orientation_datafile_list[0])
    if len(orientation_datafile_list) > 1:
        for i, orientation_datafile in enumerate(orientation_datafile_list[1:]):
            new_ufmf_data = load_ufmf_data_from_dict(orientation_datafile)
            for d, data in enumerate(ufmf_data):
                ufmf_data[d].extend( new_ufmf_data[d] )
    return ufmf_data
    
    
def save_ufmf_orientation_data_to_dataset(path):
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    
    if 1:
        culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
        dataset = fad.load(culled_dataset_filename)
    
    orientation_datafiles = [os.path.join(path, orientation_datafile) for orientation_datafile in config.orientation_datafiles] 
    print 
    print orientation_datafiles
    
    
    if 1:
        for i, orientation_datafile in enumerate(orientation_datafiles):
            ufmf_data = load_ufmf_data_from_dict(orientation_datafile)
            get_fast_heading_and_orientation(dataset, orientation_datafile=None, keys=None, ufmf_data=ufmf_data, save=True, h5=config.orientation_h5s[i])
        
        print 'SAVING culled dataset with orientation data to: ', config.path_to_culled_dataset
        dataset.save(config.path_to_culled_dataset)
    


###
def get_fast_heading_and_orientation(dataset, orientation_datafile=None, frame_to_key=None, h5=None, keys=None, ufmf_data=None, save=False):
    '''
    save    -- save orientation, eccentricity, and frame nums to trajecs. Don't return anything else
    '''

    if ufmf_data is None:
        timestamp_ufmf, longaxis, eccentricity, ufmf_frames_ufmf, center = load_ufmf_data_from_dict(orientation_datafile)
    else:
        timestamp_ufmf, longaxis, eccentricity, ufmf_frames_ufmf, center = ufmf_data
    
    print 'getting ufmf frame offset'
    ufmf_frame_offset = get_ufmf_frame_offset(dataset, ufmf_frames_ufmf, timestamp_ufmf, npts=25)
    
    if keys is None:
        keys = dataset.trajecs.keys()
        
    # first initialize all flies with orientation, eccentricity and frames attributes
    print 'initializing'
    for key in dataset.trajecs.keys():
        if h5 in key:
            trajec = dataset.trajecs[key]
            trajec.orientation = []
            trajec.eccentricity = []
            trajec.frames_with_orientation = []
            trajec.orientation_center = []
            trajec.ufmf_frame_offset = ufmf_frame_offset
            
    if frame_to_key is None:
        frame_to_key = fad.get_frame_to_key_dict(h5, dataset)
        
    for ufmf_index, camera_frame in enumerate(ufmf_frames_ufmf):
        if (camera_frame/100.) == int(camera_frame/100.):
            print camera_frame
            
        flydra_camera_frame = ufmf_frame_offset + camera_frame
            
        if eccentricity[ufmf_index] is None:
            continue
        if eccentricity[ufmf_index] > 1:
            continue
        try:
            possible_keys = frame_to_key[flydra_camera_frame]
        except:
            continue # must a frame during which there were no trajectories
        if len(possible_keys) == 0:
            continue
        keys_in_volume = get_keys_in_volume_for_camera_frame(flydra_camera_frame, frame_to_key, dataset)
        if len(keys_in_volume) < 1:
            continue # no flies
        if len(keys_in_volume) > 1:
            continue # too many flies
        key = keys_in_volume[0]
        trajec = dataset.trajecs[key]
        fly_frame = flydra_camera_frame - trajec.first_frame
            
        # fix orientation for airvelocity
        unsigned_orientation = np.arctan2(longaxis[ufmf_index][0]*-1, longaxis[ufmf_index][1]) #orientation_ufmf[ufmf_index]
        velocity_heading = trajec.airheading_smooth[fly_frame]
        orientation = unsigned_orientation
        if 1:
            #orientation = -1*unsigned_orientation
            n = 0
            while np.abs(floris_math.fix_angular_rollover(orientation-velocity_heading)) > np.pi/2.:
            #if np.abs(orientation-velocity_heading) > np.pi/2.:
                n += 1
                if n > 2:
                    print n, orientation, velocity_heading
                if 1:
                    if orientation < 0:
                        orientation += np.pi
                    else:
                        orientation -= np.pi
                
        # unwrap orientation:
        orientation = floris_math.fix_angular_rollover(orientation)
        
        
        
        # save data to trajec
        trajec.orientation.append(orientation)
        trajec.eccentricity.append(eccentricity[ufmf_index])
        trajec.frames_with_orientation.append(fly_frame)
        trajec.orientation_center.append(center[ufmf_index])
        print key
        
    return

######################################################################################################################
# Process UFMF

def extract_unsigned_orientation_and_position(img):
    center, longaxis, shortaxis, body, ratio = nim.find_ellipse(img, background=None, threshrange=[-100,-15], sizerange=[10,500], erode=2, autothreshpercentage=None)
    
    if ratio[0] is not None:
        eccentricity = ratio[1] / ratio[0]
    else:
        eccentricity = None
        
    return center, longaxis, eccentricity
    
    if 0:
        unsigned_orientation = np.arctan2(longaxis[0], longaxis[1])
        position = center[::-1]
        if ratio[0] is not None:
            eccentricity = ratio[1] / ratio[0]
        else:
            eccentricity = None
        
        return position, unsigned_orientation, eccentricity         
    

def main(filename, start=0, end=-1, frames_to_process='all'):
    orientation_frames = {}
    movie = ufmf.FlyMovieEmulator(filename)
    if end == -1:
        end = movie.get_n_frames()
    
    if frames_to_process == 'all':
        frames_to_process = np.arange(start, end).tolist()
    
    prev_pos = None
    
    #frames_to_process = get_camera_frames_with_keys_in_volume(dataset)
    for frame in frames_to_process:
        if (frame/100.) == int(frame/100.):
            print frame
        img = -1*(movie.get_mean_for_timestamp(movie.get_frame(frame)[1]) - movie.get_frame(frame)[0])
        
        if np.min(img) < -20:
            timestamp = movie.get_frame(frame)[1]
            position, longaxis, eccentricity = extract_unsigned_orientation_and_position(img)
            if eccentricity is not None:
                print frame
            
            framedata = {'frame': frame, 'timestamp': timestamp, 'longaxis': longaxis, 'eccentricity': eccentricity, 'position': position}
            orientation_frames.setdefault(frame, framedata)
        else:
            framedata = None
            orientation_frames.setdefault(frame, framedata)
        
    return orientation_frames
    
    

def process_ufmf(path, ufmf_filename, start, stop, frames_to_process=None):
    
        
    try: 
        #culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
        #dataset = fad.load(culled_dataset_filename)
        analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
        config = analysis_configuration.Config(path)
        ufmf_path = os.path.join(path, 'data', 'ufmfs')  
        frames_to_process = 'all'
        if 0:#frames_to_process is None:
            frame_to_key = fad.get_frame_to_key_dict('20121002', dataset) 
            frames_to_process = get_camera_frames_with_keys_in_volume(frame_to_key, dataset)
    except:    
        ufmf_path = ''    
        frames_to_process = 'all'
    orientation_frames = main(ufmf_filename, start, stop, frames_to_process=frames_to_process)
    
    filename = 'tmp_' + 'DATA_' + os.path.basename(ufmf_filename).split('.')[0]
    filename_with_path = os.path.join(ufmf_path, filename) 
        
    f = open(filename_with_path, 'w')
    pickle.dump(orientation_frames, f)
    f.close()

def save_ufmf_images_to_directory(ufmf_filename, img_directory, frames):
    movie = ufmf.FlyMovieEmulator(ufmf_filename)
    
    def make_str_n_long(n, nlen=6):
        while len(n) < nlen:
            n = '0' + n
        return n
    
    n = 0
    for frame in frames:
        if (frame/100.) == int(frame/100.):
            print frame
        img = -1*(movie.get_mean_for_timestamp(movie.get_frame(frame)[1]) - movie.get_frame(frame)[0])
        
        n += 1
        nstr = make_str_n_long(str(n))
        fstr = nstr+'_'+str(frame)+'.png'
        imname = os.path.join(img_directory, fstr)
        plt.imsave(imname, img)

######################################################################################################################
# Play movie
def example_movie(path, orientation_datafile, img_directory, ufmf_filename, save_movie_path='', nframes=200, firstframe=10):
    '''
    path = '/home/caveman/DATA/20120924_HCS_odor_horizon'
    orientation_datafile = '/home/caveman/DATA/tmp_orientation_checks/data.pickle'
    img_directory = '/home/caveman/DATA/tmp_orientation_checks/images'
    ufmf_filename = '/home/caveman/DATA/20120924_HCS_odor_horizon/data/ufmfs/small_20121002_184626_Basler_21111538.ufmf'
    '''
    
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
    dataset = fad.load(culled_dataset_filename)
    
    heading, orientation, eccentricity, speed, velocities, airvelocities, frames, ufmf_frames, x, y, key = get_heading_and_orientation(dataset, orientation_datafile, keys=None, ufmf_data=None)
    
    frames_to_play = np.arange(firstframe, firstframe+nframes).tolist()
    
    data = heading, orientation, eccentricity, speed, velocities, airvelocities, frames, ufmf_frames, x, y, key
    for d, datum in enumerate(data):
        new_datum = [datum[i] for i in frames_to_play]
        data[d] = new_datum
    heading, orientation, eccentricity, speed, velocities, airvelocities, frames, ufmf_frames, x, y, key = data
    
    print 'first ufmf frame: ', ufmf_frames[0]
    
    if not os.path.isdir(img_directory):
        os.mkdir(img_directory)
    save_ufmf_images_to_directory(ufmf_filename, img_directory, ufmf_frames)
    
    # check to make sure we have same number of images as data points
    images = flyanim.get_image_file_list(img_directory)
    print 'n images: ', len(images)
    print 'n data pts: ', len(x)
    assert(len(images)==len(x))
            
    # optional parameters
    color = 'none'
    edgecolor = 'red'
    ghost_tail = 20
    nskip = 0
    wedge_radius = 25
    imagecolormap = 'gray'
    
    # get x/y limits (from image)
    img = flyanim.get_nth_image_from_directory(0, img_directory)
    xlim = [0, img.shape[1]]
    ylim = [0, img.shape[0]]
    
    if len(save_movie_path) > 0:
        save = True
        if not os.path.isdir(save_movie_path):
            os.mkdir(save_movie_path)
    else:
        save = False
            
    # useful parameters for aligning image and data:
    # extent, origin, flipimgx
    
    print orientation
            
    # play the movie!
    flyanim.play_movie(x.tolist(), y.tolist(), color=color, images=img_directory, orientation=orientation, save=save, save_movie_path=save_movie_path, nskip=nskip, ghost_tail=ghost_tail, wedge_radius=wedge_radius, xlim=xlim, ylim=ylim, imagecolormap=imagecolormap, edgecolor=edgecolor, flipimgx=False, flipimgy=False, flip=False)


def trajectory_movie(path, img_directory, ufmf_filename, h5, save_movie_path='', nkeys=5):
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
    dataset = fad.load(culled_dataset_filename)
    frame_to_key = fad.get_frame_to_key_dict(h5, dataset)
    trajectory_movie_from_dataset(dataset, img_directory, frame_to_key, ufmf_filename, h5, save_movie_path, nkeys)
    
# Play movie from TRAJECS
def trajectory_movie_from_dataset(dataset, img_directory, frame_to_key, ufmf_filename, h5, save_movie_path='', nkeys=5, keys=None):
    '''
    path = '/home/caveman/DATA/20120924_HCS_odor_horizon'
    orientation_datafile = '/home/caveman/DATA/tmp_orientation_checks/data.pickle'
    img_directory = '/home/caveman/DATA/tmp_orientation_checks/images'
    ufmf_filename = '/home/caveman/DATA/20120924_HCS_odor_horizon/data/ufmfs/small_20121002_184626_Basler_21111538.ufmf'
    '''
    
    if keys is None:
        keys = dataset.trajecs.keys()

    orientation = []
    center = []
    ufmf_frames = []

    n = -1
    for key in keys:
        if h5 not in key:
            continue
        trajec = dataset.trajecs[key]
        if len(trajec.frames_with_orientation) < 1:
            continue
        print key
        n += 1
        if n >= nkeys:
            break
        print n
        orientation.extend(copy.copy(trajec.orientation))
        center.extend(copy.copy(trajec.orientation_center))
        ufmf_frames.extend( (np.array(copy.copy(trajec.frames_with_orientation)) + trajec.first_frame - trajec.ufmf_frame_offset).tolist())
    
    print 'first ufmf frame: ', ufmf_frames[0]
    print 'n frames: ', len(ufmf_frames)
    
    class ImgMovie(object):
        def __init__(self, movie, ufmf_frames):
            self.movie = movie
            self.ufmf_frames = ufmf_frames
        def get_frame(self, frame):
            ufmf_frame = self.ufmf_frames[frame]
            return -1*(self.movie.get_mean_for_timestamp(self.movie.get_frame(ufmf_frame)[1]) - self.movie.get_frame(ufmf_frame)[0])
    
    movie = ufmf.FlyMovieEmulator(ufmf_filename)
    imgmovie = ImgMovie(movie, ufmf_frames)
    images = imgmovie.get_frame
    
    if 0:
        print 'saving images'
        if not os.path.isdir(img_directory):
            os.mkdir(img_directory)
        save_ufmf_images_to_directory(ufmf_filename, img_directory, ufmf_frames)
        print 'done with saving images'
        print
    
    
    # get x/y limits (from image)
    #img = flyanim.get_nth_image_from_directory(0, img_directory)
    img = images(0)
    xlim = [0, img.shape[1]]
    ylim = [0, img.shape[0]]
    
    x = []
    y = []
    for c in center:
        x.append(c[1])
        y.append(img.shape[0] - c[0])
    
    print 'length of x: ', len(x)
    
    # check to make sure we have same number of images as data points
    #images = flyanim.get_image_file_list(img_directory)
    #print 'n images: ', len(images)
    #print 'n data pts: ', len(orientation)
    #assert(len(images)==len(orientation))
            
    # optional parameters
    color = 'none'
    edgecolor = 'red'
    ghost_tail = 20
    nskip = 0
    wedge_radius = 25
    imagecolormap = 'gray'
    
    if len(save_movie_path) > 0:
        save = True
        if not os.path.isdir(save_movie_path):
            os.mkdir(save_movie_path)
    else:
        save = False
            
    # useful parameters for aligning image and data:
    # extent, origin, flipimgx
    
    print orientation
            
    # play the movie!
    flyanim.play_movie(x, y, color=color, images=images, orientation=orientation, save=save, save_movie_path=save_movie_path, nskip=nskip, ghost_tail=ghost_tail, wedge_radius=wedge_radius, xlim=xlim, ylim=ylim, imagecolormap=imagecolormap, edgecolor=edgecolor, flipimgx=True, flipimgy=False, flip=False)

#############################################################################################################################3
# Plot trajectories

def plot_trajectory_from_path(path):
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    
    culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
    dataset = fad.load(culled_dataset_filename)
    
    plot_trajectory(dataset, config)

def plot_trajectory(dataset, config):
    path = config.path
    keys = dataset.trajecs.keys()#fad.get_keys_with_attr(dataset, ['odor_stimulus'], ['on'])

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
        ax.set_xlim(-.1, .3)
        ax.set_ylim(-.15, .15)
        ax.set_aspect('equal')
        ax.set_title(key.replace('_', '-'))
            
        ax.plot(trajec.positions[frames[0]-10:frames[-1]+10,0], trajec.positions[frames[0]-10:frames[-1]+10,1], 'black', zorder=-100, linewidth=0.25)
        
        fpl.colorline_with_heading(ax,trajec.positions[frames,0], trajec.positions[frames,1], trajec.eccentricity, orientation=trajec.orientation, colormap='jet', alpha=1, colornorm=[0,.8], size_radius=0.15-np.abs(trajec.positions[frames,2]), size_radius_range=[.02, .02], deg=False, nskip=0, center_point_size=0.01, flip=False)
            
            
        pp.savefig()
        plt.close('all')

    pp.close()
    
    
def save_raw_orientation_data_images(orientation_datafile, ufmf_filename, destination):

    timestamp, longaxis, eccentricity, ufmf_frames, center = load_ufmf_data_from_dict(orientation_datafile)
    movie = ufmf.FlyMovieEmulator(ufmf_filename)
    
    for f, frame in enumerate(ufmf_frames):
        if eccentricity[f] is None:
            print frame, ' no data'
            continue
        else:
            print frame
        
        img = -1*(movie.get_mean_for_timestamp(movie.get_frame(frame)[1]) - movie.get_frame(frame)[0])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        
        circle = patches.Circle((center[f][1], center[f][0]), 1, facecolor='white', edgecolor='none')
        ax.add_artist(circle)
        ax.plot([center[f][1]-longaxis[f][1]*5, center[f][1]+longaxis[f][1]*5], [center[f][0]-longaxis[f][0]*5, center[f][0]+longaxis[f][0]*5], zorder=10, color='white')
        
        figname = os.path.join(destination, str(f) + '_' + str(frame)) + '.png'
        fig.savefig(figname, format='png')
        plt.close('all')
            
            
            
def process_raw_orientation_data(orientation_datafiles):
    if type(orientation_datafiles) is not list:
        orientation_datafiles = [orientation_datafiles]
    timestamp, longaxis, eccentricity, ufmf_frames, center = load_ufmf_data_from_dict_list(orientation_datafiles)
    timestamp = np.array(timestamp)
    eccentricity = np.array(eccentricity)
    
    bad_measurements = []
    for f, eccen in enumerate(eccentricity):
        if eccen is None:
            bad_measurements.append(f)
    ufmf_frames = np.array(ufmf_frames)
    ufmf_frames[bad_measurements] = -1
    ufmf_frames = ufmf_frames.tolist()
    
    chunks, break_points = flymath.get_continuous_chunks(ufmf_frames)
    
    body_trajectories = {}
    
    key = -1
    for i, break_point in enumerate(break_points):
        key += 1
        if break_point >= len(ufmf_frames):
            break
        chunk = ufmf_frames[break_point:break_points[i+1]]
        if type(chunk) is not list:
            chunk = chunk.tolist()
            
        if len(chunk) < 10:
            continue
        print key
            
        heading = np.zeros(len(chunk))
        heading_air = np.zeros(len(chunk))
        orientation = np.zeros(len(chunk))
        position = np.zeros([len(chunk), 2])
        speed = np.zeros(len(chunk))
        
        #chunk_indices = np.arange(0, len(chunk))
        #ufmf_indices = np.arange(break_point,break_points[i+1])
        
        #vel_norm = [(center[b] - center[b-1]/np.linalg.norm(center[b]
        for chunk_index, b in enumerate(np.arange(break_point,break_points[i+1])):
            if chunk_index == 0:
                continue
            vel_norm = (center[b] - center[b-1])
            speed[chunk_index] = np.linalg.norm(copy.copy(vel_norm))
            vel_norm = vel_norm / np.linalg.norm(vel_norm)
            vel_norm_air = copy.copy(vel_norm)
            vel_norm_air[1] -= 0.3
            vel_norm_air = vel_norm / np.linalg.norm(vel_norm)
            heading[chunk_index] = np.arctan2(vel_norm[0]*-1, vel_norm[1])
            heading_air[chunk_index] = np.arctan2(vel_norm_air[0]*-1, vel_norm[1])
            
            if np.isnan(heading[chunk_index]):
                bad_data = True
                break
            else:
                bad_data = False
            
            # flip heading
            if heading_air[chunk_index] < 0:
                heading_air[chunk_index] += np.pi
            else:
                heading_air[chunk_index] -= np.pi
            
            if heading[chunk_index] < 0:
                heading[chunk_index] += np.pi
            else:
                heading[chunk_index] -= np.pi
                    
            position[chunk_index] = center[b]
            
            orientation[chunk_index] = np.arctan2(longaxis[b][0]*-1, longaxis[b][1])
            n = 0
            if 1:
                while np.abs(flymath.fix_angular_rollover(orientation[chunk_index]-orientation[chunk_index-1])) > np.pi/2.:
                    n += 1
                    if orientation[chunk_index] < 0:
                        orientation[chunk_index] += np.pi
                    else:
                        orientation[chunk_index] -= np.pi
                
                while np.abs(flymath.fix_angular_rollover(orientation[chunk_index]-heading_air[chunk_index])) > np.pi*0.7:
                    n += 1
                    if orientation[chunk_index] < 0:
                        orientation[chunk_index] += np.pi
                    else:
                        orientation[chunk_index] -= np.pi
        
        if bad_data is True:
            continue
            
        heading[0] = heading[1]
        heading_air[0] = heading_air[1]
        position[0] = center[break_point]
        orientation[0] = np.arctan2(longaxis[break_point][0]*-1, longaxis[break_point][1])
        n = 0
        while np.abs(flymath.fix_angular_rollover(orientation[0]-orientation[1])) > np.pi/2.:
        #if np.abs(orientation-velocity_heading) > np.pi/2.:
            n += 1
            if orientation[0] < 0:
                orientation[0] += np.pi
            else:
                orientation[0] -= np.pi
                
        ######### 
        # smooth
        data = flymath.remove_angular_rollover(orientation, 3)
        data = data.reshape([len(data),1])
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

        orientation_smooth = flymath.fix_angular_rollover(xsmooth[:,0])        
        slip_angles = np.abs(flymath.fix_angular_rollover(orientation_smooth - heading_air))
        if np.mean(slip_angles) > np.pi/2.: # flip all the orientations
            pos = np.where(orientation_smooth > 0)
            neg = np.where(orientation_smooth > 0)
            orientation_smooth[pos] -= np.pi
            orientation_smooth[neg] += np.pi
        
        
        trajec = {}
        trajec.setdefault('ufmf_timestamps', timestamp[break_point:break_points[i+1]])
        trajec.setdefault('ufmf_frames', chunk)
        trajec.setdefault('eccentricity', eccentricity[break_point:break_points[i+1]])
        trajec.setdefault('heading', heading_air)
        trajec.setdefault('heading', heading)
        trajec.setdefault('speed', speed)
        trajec.setdefault('orientation', orientation)
        trajec.setdefault('orientation_smooth', orientation_smooth)
        trajec.setdefault('position', position)
        trajec.setdefault('key', key)
        body_trajectories.setdefault(key, trajec)
    
    return body_trajectories
    
def bind_body_trajectories_to_dataset(config, dataset, orientation_datafiles, ufmf_frame_offset=None, body_keys=None):
    # note: orientation_datafile must contain something like this: DATA_small_20121002_184626_Basler_21111538
    
    for n, orientation_datafile in enumerate(orientation_datafiles):
        # get date
        text_string = 'DATA_small_'
        start = orientation_datafile.index(text_string)
        start_of_date = start+len(text_string)
        h5_date = orientation_datafile[start_of_date:start_of_date+8]
        start = orientation_datafile.index('Basler')
        cam_id = orientation_datafile[start:]
        ufmf_frame_offset = None
        body_keys = None

        # collect data    
        frame_to_key = fad.get_frame_to_key_dict(h5_date, dataset)
        body_trajectories = process_raw_orientation_data(orientation_datafile)

        # get ufmf_frame_offset    
        if ufmf_frame_offset is None:
            ufmf_frames = []
            ufmf_timestamps = []
            while len(ufmf_frames) < 200:
                for body_key, body_trajec in body_trajectories.items():
                    ufmf_frames.extend(body_trajec['ufmf_frames'])
                    ufmf_timestamps.extend(body_trajec['ufmf_timestamps'])
            print 'getting ufmf frame offset'
            ufmf_frame_offset = get_ufmf_frame_offset(dataset, ufmf_frames, ufmf_timestamps, npts=25)
            del(ufmf_frames)
            del(ufmf_timestamps)
            
        # clear old data
        for key, trajec in dataset.trajecs.items():
            if h5_date in key:
                trajec.frames_with_orientation_data = []
                trajec.body_orientation_data = []

        # associate body_trajecs with flydra trajecs
        print 'associating body trajecs with flydra trajecs'
        if body_keys is None:
            body_keys = body_trajectories.keys()
        elif type(body_keys) is not list:
            body_keys = [body_keys]
        print 'N body keys: ', len(body_keys)
        for body_key in body_keys:
            body_trajec = body_trajectories[body_key]
            first_ufmf_frame = body_trajec['ufmf_frames'][0]
            # frame_offset = flydra_frame - ufmf_frame
            flydra_frame = first_ufmf_frame + ufmf_frame_offset
            try:
                keys = get_keys_in_volume_for_flydra_frame(flydra_frame, frame_to_key, dataset)
            except:
                keys = []
                print 'no key!'
            if len(keys) == 1:
                trajec = dataset.trajecs[keys[0]]
                trajec.body_orientation_data.append(body_trajec)
                trajec.ufmf_frame_offset = copy.copy(ufmf_frame_offset)
                flydra_frames = np.array(body_trajec['ufmf_frames']) + ufmf_frame_offset
                fly_frames = flydra_frames - trajec.first_frame
                trajec.frames_with_orientation_data.append(fly_frames)
                print 'set orientations for: ', keys[0]

def find_trajec_with_body_orientation_key(dataset, body_key):
    for key, trajec in dataset.trajecs.items():
        try:
            body_keys = []
            for body_trajec in trajec.body_orientation_data:
                body_keys.append(body_trajec['key'])
            if body_key in body_keys:
                return key
        except:
            pass
            

def get_body_trajectories_from_dataset(dataset, keys=None, odor=True, threshold_odor=10, history=300):
    if keys is None:
        keys = dataset.trajecs.keys()

    body_trajectories = {}
    n = 0
    for key in keys:
        trajec = dataset.trajecs[key]
        
        try:
            body_data = trajec.body_orientation_data
        except:
            body_data = []
        if len(body_data) < 1:
            continue
            
        for i, data in enumerate(body_data):
            
            if 0:
                if odor: # select trajecs that come history frames after odor
                    fwod = trajec.frames_with_orientation_data[i]
                    frames_start = fwod[0] - history
                    frames_start = np.max([frames_start, 0])
                    frames_end = np.min([fwod[-1], trajec.length])
                    frames = np.arange(frames_start, fwod[0])
                    if np.max(trajec.odor[frames]) > threshold_odor:
                        pass
                    else:
                        continue
                else:
                    fwod = trajec.frames_with_orientation_data[i]
                    frames_start = fwod[0] - history
                    frames_start = np.max([frames_start, 0])
                    frames_end = np.min([fwod[-1], trajec.length])
                    frames = np.arange(0, frames_end)
                    if np.max(trajec.odor[frames]) > threshold_odor:
                        continue
                    else:
                        pass
            
                    
            try:
                data.setdefault('speed', 0)
                data['speed'] = trajec.speed_xy[trajec.frames_with_orientation_data[i]]
                n += 1
                body_trajectories.setdefault(n, data)
            except:
                pass
                
    return body_trajectories
    
def plot_heading_vs_orientation_trajectories(body_trajectories, config, keys=None):
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'body_orientation_trajectories_test.pdf')
    pp = PdfPages(pdf_name_with_path)
    
    if keys is None:
        keys = body_trajectories.keys()
        
    n = 0
    for key in keys:
        trajec = body_trajectories[key]
        
        # ignore trajectories where mean slip angle is less than X
        indices_crosswind = np.where( (np.abs(trajec['heading']) > 45*np.pi/180.)*(np.abs(trajec['heading']) < 135*np.pi/180.) )[0]
        slipangles = (trajec['heading'] - trajec['orientation_smooth'])[indices_crosswind]
        if np.mean(np.abs(slipangles)) < 45*np.pi/180.:
            pass
        else:
            continue
        
        if 1: #(trajec['position'][-1,1] - trajec['position'][0,1]) < 0:
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111)
        
            try:
            
                fpl.colorline_with_heading(ax, trajec['position'][:,1], -1*trajec['position'][:,0], 'red', orientation=trajec['heading'], size_radius=15, deg=False, nskip=0, center_point_size=0.01, flip=False)
                
                fpl.colorline_with_heading(ax, trajec['position'][:,1], -1*trajec['position'][:,0], 'black', orientation=trajec['orientation_smooth'], size_radius=15, deg=False, nskip=0, center_point_size=0.01, flip=False)
                
                ax.set_xlim(0,650)
                ax.set_ylim(-450, 0)
                
                        
                ax.set_aspect('equal')
                ax.set_title(str(key))
                
                pp.savefig()
                plt.close('all')
                
                n += 1
            except:
                print key, ' error!'
    pp.close()
    
    

def plot_heading_vs_orientation_scatter(body_trajectories, config, keys=None):
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'body_orientation_trajectories_scatter_odor_on.pdf')
    
    if keys is None:
        keys = body_trajectories.keys()
    
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    
    headings = []
    orientations = []
    speeds = []
    for key in keys:
        trajec = body_trajectories[key]
        
        if np.max(np.isnan(trajec['heading'])):
            print trajec['heading']
            break
            
        #if trajec['position'][-1,1] > trajec['position'][0,1]:
        #    continue
            
        headings.extend( (trajec['heading']*180./np.pi).tolist() )
        orientations.extend ( (trajec['orientation_smooth']*180./np.pi).tolist() )
        speeds.extend ( np.abs(trajec['speed']).tolist() )

    indices_with_low_speed = np.where(np.array(speeds) < 0.3)[0]
    indices_with_high_speed = np.where(np.array(speeds) > 0.3)[0]
    
    indices = indices_with_low_speed

    ax.set_aspect('equal')
    #fpl.scatter(ax, np.array(headings)[indices], np.array(orientations)[indices], color=np.array(speeds)[indices], radius=0.5, colornorm=[0,.6])
    fpl.scatter(ax, np.array(headings), np.array(orientations), color=np.array(speeds), radius=0.5, colornorm=[0,.6])
    xticks = [-180, -90, 0, 90, 180]
    yticks = [-180, -90, 0, 90, 180]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks)
    ax.set_xlabel('Heading (relative to ground)')
    ax.set_ylabel('Body angle')
    ticklabels = ['-180', '-90', 'upwind', '90', '180']
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    
    
    fig.savefig(pdf_name_with_path, format='pdf')        
    
    
# difference between odor and no odor
def get_orientations_and_headings_for_body_trajectories(body_trajectories):
    keys = body_trajectories.keys()
    headings = []
    orientations = []
    speeds = []
    for key in keys:
        trajec = body_trajectories[key]
        
        if np.max(np.isnan(trajec['heading'])):
            print trajec['heading']
            break
            
        #if trajec['position'][-1,1] > trajec['position'][0,1]:
        #    continue
            
        headings.extend( (trajec['heading']*180./np.pi).tolist() )
        orientations.extend ( (trajec['orientation_smooth']*180./np.pi).tolist() )
        speeds.extend ( np.abs(trajec['speed']).tolist() )
    return np.array(headings), np.array(orientations), np.array(speeds)
    
def plot_hist(ax, hist, x, y, logcolorscale=True, colornorm=None):
    if logcolorscale:
        hist = np.log(hist+1) # the plus one solves bin=0 issues

    if 0:
        if colornorm is not None:
            colornorm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
        else:
            colornorm = matplotlib.colors.Normalize(np.min(np.min(hist)), np.max(np.max(hist)))
    
    xextent = [x[0], x[-1]]
    yextent = [y[0], y[-1]]
        
    # make the heatmap
    cmap = plt.get_cmap('jet')
    ax.imshow(  hist.T, 
                cmap=cmap,
                extent=(xextent[0], xextent[1], yextent[0], yextent[1]), 
                origin='lower', 
                interpolation='nearest',
                #norm=colornorm,
                )
    ax.set_aspect('auto')
    
    
def show_orientation_difference(dataset, config):

    keys_odor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'on')
    keys_noodor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'none')
    keys_afterodor = fad.get_keys_with_attr(dataset, 'odor_stimulus', 'afterodor')
    keys_noodor.extend(keys_afterodor)
    
    body_trajectories_odor = get_body_trajectories_from_dataset(dataset, keys=keys_odor)
    body_trajectories_noodor = get_body_trajectories_from_dataset(dataset, keys=keys_noodor)
    
    if len(body_trajectories_noodor) < len(body_trajectories_odor):
        keys = body_trajectories_odor.keys()[0:len(body_trajectories_noodor)]
        body_trajectories_odor_reduced = {}
        for key in keys:
            body_trajectories_odor_reduced.setdefault(key, body_trajectories_odor[key])
        body_trajectories_odor = body_trajectories_odor_reduced
    
    headings_odor, orientations_odor, speeds_odor = get_orientations_and_headings_for_body_trajectories(body_trajectories_odor)
    headings_noodor, orientations_noodor, speeds_noodor = get_orientations_and_headings_for_body_trajectories(body_trajectories_noodor)

    binsx = np.linspace(-180,180,50)
    binsy = np.linspace(-180,180,50)

    hist_odor,x,y = np.histogram2d(headings_odor, orientations_odor, (binsx,binsy), normed=True)
    hist_noodor,x,y = np.histogram2d(headings_noodor, orientations_noodor, (binsx,binsy), normed=True)
    
    hist_odor -= np.min(hist_odor)
    hist_odor /= np.max(hist_odor)
    
    hist_noodor -= np.min(hist_noodor)
    hist_noodor /= np.max(hist_noodor)
    
    hist_diff = hist_odor-hist_noodor
    hist_diff -= np.min(hist_diff)
    hist_diff /= np.max(hist_diff)
    
    fig = plt.figure(figsize=(10,4))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    axodor = fig.add_subplot(131)
    axnoodor = fig.add_subplot(132)
    axdiff = fig.add_subplot(133)
    
    plot_hist(axodor, hist_odor, x, y, logcolorscale=True)
    plot_hist(axnoodor, hist_noodor, x, y, logcolorscale=True)
    plot_hist(axdiff, hist_diff, x, y, logcolorscale=True)
    
    axes = [axodor, axnoodor, axdiff]
    
    for ax in axes:
        ax.set_aspect('equal')
    
    ticks = [-180, -90, 0, 90, 180]
    ticklabels = ['-180', '-90', 'upwind', '90', '180']
    fpl.adjust_spines(axodor, ['left', 'bottom'], xticks=ticks, yticks=ticks)
    axodor.set_xticklabels(ticklabels)
    axodor.set_xlabel('heading, deg')
    axodor.set_yticklabels(ticklabels)
    axodor.set_ylabel('body orientation, deg')
    axodor.set_title('Odor present')
    
    fpl.adjust_spines(axnoodor, ['bottom'], xticks=ticks, yticks=ticks)
    axnoodor.set_xticklabels(ticklabels)
    axnoodor.set_xlabel('heading, deg')
    axnoodor.set_title('No odor present')
    
    fpl.adjust_spines(axdiff, ['bottom'], xticks=ticks, yticks=ticks)
    axdiff.set_xticklabels(ticklabels)
    axdiff.set_xlabel('heading, deg')
    axdiff.set_title('Change in behavior due to odor')

    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'orientation_odor_no_odor_difference.pdf')
    fig.savefig(pdf_name_with_path, format='pdf')     
    

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
    parser.add_option("--img_directory", type="str", dest="img_directory", default='',
                        help="path where save ufmf images for movie") 
    parser.add_option("--save_movie_path", type="str", dest="save_movie_path", default='',
                        help="path to where tmp images should be save for animation movie")
    parser.add_option("--nkeys", type=int, dest="nkeys", default=5,
                        help="how many keys for movie?")
    parser.add_option("--h5", type="str", dest="h5", default='',
                        help="h5 date number to work with")
    
                        
    (options, args) = parser.parse_args()
    
    print 
    print 'running: ', options.action
    
    if options.action == 'process_ufmf':
        process_ufmf(options.path, options.file, options.start, options.stop)
    elif options.action == 'play_movie':
        # needs: path, img_directory, ufmf_filename, save_movie_path, play_movie
        #example_movie(options.path, options.orientation, options.img_directory, options.file, options.save_movie_path)
        trajectory_movie(options.path, options.img_directory, options.file, options.save_movie_path, nkeys=5)
    elif options.action == 'save':
        print 'PATH: ', options.path
        save_ufmf_orientation_data_to_dataset(options.path)
    elif options.action == 'trajectory_movie':
        trajectory_movie(options.path, options.img_directory, options.file, options.h5, options.save_movie_path, nkeys=options.nkeys)
