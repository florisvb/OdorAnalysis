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

##########################################################################################################
# load and interpret data

def get_ufmf_frame_offset(dataset, ufmf_frames, ufmf_timestamps, npts=25):
    # find frames where difference < 0.008
    frame_offsets = []
    
    n = 0
    for key in dataset.trajecs.keys():
        trajec = dataset.trajecs[key]
        for f, t in enumerate(trajec.timestamp_epoch + trajec.time_fly):
            tdiff = np.abs(t - np.array(ufmf_timestamps))
            if np.min(tdiff) < 0.008:
                n += 1
                if n > npts:
                    return int(np.mean(np.array(frame_offsets)))
                    #return frame_offsets
                ufmf_timestamps_index = np.argmin(tdiff)
                ufmf_frame = ufmf_frames[ufmf_timestamps_index]
                frame_offset = trajec.first_frame + f - ufmf_frame
                frame_offsets.append(frame_offset)
                break # only one data point per fly
    return int(np.mean(np.array(frame_offsets)))
    

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
    
def get_keys_in_volume_for_camera_frame(camera_frame, frame_to_key, dataset):
            
    keys = frame_to_key[camera_frame]
    keys_in_volume = []
    for key in keys:
        trajec = dataset.trajecs[key]
        camera_frames = (trajec.first_frame + np.arange(0, trajec.length)).tolist()
        index = camera_frames.index(camera_frame)
        pos = trajec.positions[index]
        in_volume = is_position_in_volume(pos, [-.1,.2], [-.15,.15], [-.1,.1])
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
    orientation = []
    eccentricity = []
    ufmf_frames = []
    x = []
    y = []
    for frame in data.keys():
        framedata = data[frame]
        if framedata is not None:
            timestamp.append(framedata['timestamp'])
            orientation.append(framedata['orientation'])
            eccentricity.append(framedata['eccentricity'])
            ufmf_frames.append(frame)
            x.append(framedata['position'][0])
            y.append(framedata['position'][1])
            
    timestamp_ufmf = np.array(timestamp)
    orientation_ufmf = np.array(orientation)
    eccentricity_ufmf = np.array(eccentricity)
    ufmf_frames_ufmf = ufmf_frames
    x = np.array(x)
    y = np.array(y)
    return timestamp_ufmf, orientation_ufmf, eccentricity_ufmf, ufmf_frames_ufmf, x, y
    
    
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
    culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
    dataset = fad.load(culled_dataset_filename)
    
    orientation_datafiles = [os.path.join(path, orientation_datafile) for orientation_datafile in config.orientation_datafiles] 
    
    ufmf_data = load_ufmf_data_from_dict_list(orientation_datafiles)
    
    get_fast_heading_and_orientation(dataset, orientation_datafile=None, keys=None, ufmf_data=ufmf_data, save=True)
    
    print 'SAVING culled dataset with orientation data to: ', config.path_to_culled_dataset
    dataset.save(config.path_to_culled_dataset)
    


###
def get_fast_heading_and_orientation(dataset, orientation_datafile=None, frame_to_key=None, h5=None, keys=None, ufmf_data=None, save=False):
    '''
    save    -- save orientation, eccentricity, and frame nums to trajecs. Don't return anything else
    '''

    if ufmf_data is None:
        timestamp_ufmf, orientation_ufmf, eccentricity_ufmf, ufmf_frames_ufmf, x, y = load_ufmf_data_from_dict(orientation_datafile)
    else:
        timestamp_ufmf, orientation_ufmf, eccentricity_ufmf, ufmf_frames_ufmf, x, y = ufmf_data
    
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
            trajec.orientation_x = []
            trajec.orientation_y = []
            trajec.ufmf_frame_offset = ufmf_frame_offset
            
    if frame_to_key is None:
        frame_to_key = fad.get_frame_to_key_dict(h5, dataset)
        
    for ufmf_index, camera_frame in enumerate(ufmf_frames_ufmf):
        if (camera_frame/100.) == int(camera_frame/100.):
            print camera_frame
            
        flydra_camera_frame = ufmf_frame_offset + camera_frame
            
        if eccentricity_ufmf[ufmf_index] is None:
            continue
        if eccentricity_ufmf[ufmf_index] > 1:
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
        unsigned_orientation = orientation_ufmf[ufmf_index]
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
        trajec.eccentricity.append(eccentricity_ufmf[ufmf_index])
        trajec.frames_with_orientation.append(fly_frame)
        trajec.orientation_x.append(x[ufmf_index])
        trajec.orientation_y.append(y[ufmf_index])
        
    return

######################################################################################################################
# Process UFMF

def extract_unsigned_orientation_and_position(img):
    center, longaxis, shortaxis, body, ratio = nim.find_ellipse(img, background=None, threshrange=[-100,-7], sizerange=[10,500], erode=2, autothreshpercentage=None)
    unsigned_orientation = np.arctan2(longaxis[0], longaxis[1])
    position = center[::-1]
    if ratio[0] is not None:
        eccentricity = ratio[1] / ratio[0]
    else:
        eccentricity = None
    
    return position, unsigned_orientation, eccentricity         
    

def main(filename, start=0, end=-1, saveimages=None, frames_to_process='all'):#='/home/caveman/DATA/tmp_orientation_checks/images'):
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
        
        if saveimages is not None:
            fstr = str(frame)+'.png'
            imname = os.path.join(saveimages, fstr)
            plt.imsave(imname, img)
            
        if np.min(img) < -7:
            timestamp = movie.get_frame(frame)[1]
            prev_pos, orientation, eccentricity = extract_unsigned_orientation_and_position(img)
            if orientation is not None:
                print frame, orientation, eccentricity
            
            framedata = {'frame': frame, 'timestamp': timestamp, 'orientation': orientation, 'eccentricity': eccentricity, 'position': prev_pos}
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
    
    def make_str_n_long(n, nlen=4):
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


def trajectory_movie(path, img_directory, ufmf_filename, save_movie_path='', nkeys=5):
    analysis_configuration = imp.load_source('analysis_configuration', os.path.join(path, 'analysis_configuration.py'))
    config = analysis_configuration.Config(path)
    culled_dataset_filename = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name) 
    dataset = fad.load(culled_dataset_filename)
    trajectory_movie_from_dataset(dataset, img_directory, ufmf_filename, save_movie_path, nkeys)
    
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
    x = []
    y = []
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
        orientation.extend(trajec.orientation)
        x.extend(trajec.orientation_x)
        y.extend(trajec.orientation_y)
        ufmf_frames.extend( (np.array(trajec.frames_with_orientation) + trajec.first_frame - trajec.ufmf_frame_offset).tolist())
    
    print 'first ufmf frame: ', ufmf_frames[0]
    print 'n frames: ', len(ufmf_frames)
    
    print 'saving images'
    if not os.path.isdir(img_directory):
        os.mkdir(img_directory)
    save_ufmf_images_to_directory(ufmf_filename, img_directory, ufmf_frames)
    print 'done with saving images'
    print
    
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
    flyanim.play_movie(x, y, color=color, images=img_directory, orientation=orientation, save=save, save_movie_path=save_movie_path, nskip=nskip, ghost_tail=ghost_tail, wedge_radius=wedge_radius, xlim=xlim, ylim=ylim, imagecolormap=imagecolormap, edgecolor=edgecolor, flipimgx=False, flipimgy=False, flip=False)

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
    
                        
    (options, args) = parser.parse_args()
    
    if options.action == 'process_ufmf':
        process_ufmf(options.path, options.file, options.start, options.stop)
    elif options.action == 'play_movie':
        # needs: path, img_directory, ufmf_filename, save_movie_path, play_movie
        #example_movie(options.path, options.orientation, options.img_directory, options.file, options.save_movie_path)
        trajectory_movie(options.path, options.img_directory, options.file, options.save_movie_path, nkeys=5)
    elif options.action == 'save':
        save_ufmf_orientation_data_to_dataset(options.path)
    
