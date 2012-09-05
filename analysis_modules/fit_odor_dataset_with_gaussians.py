import odor_dataset as od
import data_fit

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_positions(odor_dataset):
    x = []
    y = []
    z = []
    for key, odor_trace in odor_dataset.odor_traces.items():
        x.append( odor_trace.position[0] )
        y.append( odor_trace.position[1] )
        z.append( odor_trace.position[2] )
    return [np.array(x), np.array(y), np.array(z)]        

def get_odor(timestamps, odor_dataset):
    odor = [] # list of odor values at each PID point, for each time point
    for t in timestamps:
        odor_at_t = []
        for key, odor_trace in odor_dataset.odor_traces.items():
            odor_at_t_interpolated = np.interp(t, odor_trace.timestamps, odor_trace.voltage)
            odor_at_t.append( odor_at_t_interpolated )
        odor.append(np.array(odor_at_t))
    return odor
    
def get_odor_static(timestamps, odor_dataset):
    odor = []
    for t in timestamps:
        odor_at_t = []
        for key, odor_trace in odor_dataset.odor_traces.items():
            odor_at_t_interpolated = np.interp(t, odor_trace.timestamps, odor_trace.voltage)
            odor_at_t.append( odor_at_t_interpolated )
        odor.append(np.array(odor_at_t))
        
    # reformat odor and pull out max
    odor_array = np.zeros([len(odor), len(odor[0])])
    for i, timepoint in enumerate(odor):
        odor_array[i,:] = timepoint    
    max_odor = odor_array.max(axis=0)
        
    return max_odor

def fit_gaussian3d_timevarying(odor_dataset, timestamps=None):   

    if timestamps is None:
        timestamps = np.arange(11, 13.5, .01)
    
    positions = get_positions(odor_dataset)
    odor = get_odor(timestamps, odor_dataset)
            
    gm = data_fit.models.GaussianModel3D_TimeVarying()
    gm.fit(timestamps, odor, positions)
    
    return gm    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
