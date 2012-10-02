import sys, os
from optparse import OptionParser

import numpy as np

from flydra_analysis_tools import numpyimgproc as nim

import motmot.ufmf.ufmf as ufmf

import matplotlib.pyplot as plt

import pickle


def load_data(datafile):
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
        else:
            timestamps.append(0)
            x.append(0)
            y.append(0)
            orientation.append(0)
            eccentricity.append(0)
    
    return timestamps,x,y,orientation,eccentricity


def extract_unsigned_orientation_and_position(img):
    center, longaxis, shortaxis, body, ratio = nim.find_ellipse(img, background=None, threshrange=[0,150], sizerange=[10,350], erode=False)
    unsigned_orientation = np.arctan2(longaxis[1], longaxis[0])
    position = center
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
    orientation = unsigned_orientation
        
    return position, orientation, eccentricity
    
    


def main(filename, start, end, saveimages=None):
    orientation_frames = {}
    movie = ufmf.FlyMovieEmulator(filename)
    
    prev_pos = None
    for frame in range(start, end):
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



















if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--file", type="str", dest="file", default='',
                        help="path to ufmf you wish to process")
    parser.add_option("--start", type=int, dest="start", default=0,
                        help="first frame")
    parser.add_option("--stop", type=int, dest="stop", default=-1,
                        help="last frame")
    (options, args) = parser.parse_args()
    
    
    orientation_frames = main(options.file, options.start, options.stop)
    
    f = open('orientation_frames', 'w')
    pickle.dump(orientation_frames, f)
    f.close()
    
    
