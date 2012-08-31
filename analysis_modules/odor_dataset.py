import pickle
import numpy as np
import matplotlib.pyplot as plt
import data_fit

class Dataset():
    def __init__(self):
        self.odor_traces = {}
    def get_odor_trace(self, n=0):
        key = self.odor_traces.keys()[n] 
        return self.odor_traces[key]
        
class Odor_Trace(object):
    def __init__(self):
        self.position = None
        self.voltage = None
        self.timestamps = None
        
        # for real data will need to interpolate to nicer values
        self.raw_trace = None
        self.raw_timestamps = None
        
def get_traces_along_axis(dataset, axis, position, max_error=0.01):
    # find all 

    keys = []    
    for key, odor_trace in dataset.odor_traces.items():
        err = np.abs(odor_trace.position - position)
        err[axis] = 0
        errsum = np.sum(err)
        if errsum < max_error:
            keys.append(key)
    
    return keys
    
def calc_peak_of_odor_trace(odor_trace):
    odor_trace.peak_frame = np.argmax(odor_trace.trace)
    odor_trace.peak_time = odor_trace.timestamps[odor_trace.peak_frame]
    
def prep_data(dataset):
    for key, odor_trace in dataset.odor_traces.items():
        calc_peak_of_odor_trace(odor_trace)
        
def save_dataset(dataset, name):
    fd = open(name, 'w')
    pickle.dump(dataset, fd)
    fd.close()
    
def load_dataset(filename):
    fd = open(filename, 'r')
    dataset = pickle.load(fd)
    fd.close()
    return dataset
    
def calc_windspeed(dataset, axis=0, position=[0,.16], max_error=0.001):
    keys = get_traces_along_axis(dataset, axis, position, max_error)
    
    peak_times = []
    positions = []
    
    for key in keys:
        odor_trace = dataset.odor_traces[key]
        calc_peak_of_odor_trace(odor_trace)
        peak_times.append(odor_trace.peak_time)
        positions.append(odor_trace.position[axis])
        
    peak_times = np.array(peak_times)
    positions = np.array(positions)
    
    linearmodel = data_fit.models.LinearModel()
    linearmodel.fit(positions, peak_times, method='optimize')
    slope = linearmodel.parameters['slope']
    intercept = linearmodel.parameters['intercept']
    
    dataset.windspeed = slope
    
    print 'Wind Speed: ', dataset.windspeed
    
    return positions, peak_times
    
    
def show_positions(dataset, keys=None):
    if keys is None:
        keys = dataset.odor_traces.keys()
    
    for key in keys:
        odor_trace = dataset.odor_traces[key]
        print key, odor_trace.position
    
    
    
def plot_odor_traces(dataset, keys=None):
    
    if keys is None:
        keys = dataset.odor_traces.keys()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for key in keys:
        odor_trace = dataset.odor_traces[key]
        ax.plot(odor_trace.timestamps, odor_trace.voltage)
        
    plt.show()
    
    
def find_sets(odor_dataset, axis=0, min_keys_in_set=3, y_threshold=0.002, z_threshold=0.007, x_threshold=0.001):
    
    keys = []
    positions = []
    keys_to_positions = {}
    for key, odor_trace in odor_dataset.odor_traces.items():
        keys.append(key)
        positions.append(odor_trace.position)
        keys_to_positions.setdefault(key, odor_trace.position)
        
    def in_set(p1, p2, axis, y_threshold, z_threshold, x_threshold):
        diff = np.abs(p1 - p2)
        diff[axis] = 0
        if diff[0] < x_threshold:
            if diff[1] < y_threshold:
                if diff[2] < z_threshold:
                    return True
        return False
            
        
    sets_keys = {}
    for n, key in enumerate(keys[::-1]):
        keys_in_set = []
        #print
        #print '*** ', n, ' ***'
        
        # first make sure this position isn't already in a set
        key_is_already_in_set = False
        for s, set_contents in sets_keys.items():
            for key_in_set in set_contents:
                if key == key_in_set:
                    key_is_already_in_set = True
        if key_is_already_in_set:
            continue
            
        for n_r, key_r in enumerate(keys):
            if len(keys_in_set) > 1:
                mean_pt = np.mean(np.asarray([keys_to_positions[s] for s in keys_in_set]), axis=0)
            else:
                mean_pt = keys_to_positions[key]
            if in_set(mean_pt, keys_to_positions[key_r], axis, y_threshold, z_threshold, x_threshold):
                #print n_r, keys_to_positions[key_r], keys[n_r]
                keys_in_set.append(key_r)
                
        if len(keys_in_set) >= min_keys_in_set:
            sets_keys.setdefault(str(n), keys_in_set)
        
    return sets_keys, keys_to_positions
        
        
        
    
    
    
    
    
    
    
    
    
    

