import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import copy

import data_fit
import odor_dataset as od
import fly_plot_lib.plot as fpl

import matplotlib

def play_movie_from_model(gm=None):

    fig = plt.figure()
    t_start = 13
    anim_params = {'t': t_start, 'xlim': [0,1], 'ylim': [-.1,.1], 't_max': 19, 'dt': 0.05, 'resolution': 0.001}
    
    array, extent = gm.get_array_2d(anim_params['t'], anim_params['xlim'], anim_params['ylim'], anim_params['resolution'])
    im = plt.imshow( array, cmap=plt.get_cmap('jet'))
    

    def updatefig(*args):
        anim_params['t'] += anim_params['dt']
        if anim_params['t'] > anim_params['t_max']:
            anim_params['t'] = 10
                        
        array, extent = gm.get_array_2d(anim_params['t'], anim_params['xlim'], anim_params['ylim'], anim_params['resolution'])
        
        im.set_array(array)
        return im,

    ani = animation.FuncAnimation(fig, updatefig, anim_params, interval=50, blit=True)
    plt.show()
    

def make_false_odor_trace(gm, timestamps, position):
    
    x = position[0]
    y = position[1]
    inputs = [timestamps, [x, y]] 
    trace = gm.get_val(inputs)
    
    odor_trace = od.Odor_Trace(position)
    odor_trace.trace = trace
    odor_trace.timestamps = timestamps
    
    return odor_trace
    

def make_false_odor_dataset(gm=None, timestamps=None, positions=None):
    
    if gm is None:
        parameters = {  'mean_0_intercept': 0,
                        'mean_0_slope':     .2,
                        'mean_1_intercept': 0.16,
                        'mean_1_slope':     0,
                        'std_0_intercept':  0.2,
                        'std_0_slope':      0.05,
                        'std_1_intercept':  0.05,
                        'std_1_slope':      0.02,
                        'magnitude':        1,
                        }
        gm = data_fit.models.GaussianModel2D_TimeVarying(parameters=parameters)
    
        
    if timestamps is None:
        t_max = 3
        dt = 0.002
        timestamps = np.arange(0,t_max,dt)
    
    if positions is None:
        if 0:
            positions = [[0, .165], [.1, .165], [.2, .165], [.3, .165], [.4, .165], [.5, .165], [.6, .165], 
                         [0, .175], [.1, .175], [.2, .175], [.3, .175], [.4, .175], [.5, .175], [.6, .175],
                         [0, .135], [.1, .135], [.2, .135], [.3, .135], [.4, .135], [.5, .135], [.6, .135],
                         [0, .195], [.1, .195], [.2, .195], [.3, .195], [.4, .195], [.5, .195], [.6, .195],
                         [0, .155], [.1, .155], [.2, .155], [.3, .155], [.4, .155], [.5, .155], [.6, .155]]
        if 1:
            positions = []
            x_pos = np.arange(0,1,.05).tolist()
            y_pos = np.arange(0, .33, .05).tolist()
            for i, x in enumerate(x_pos):
                for j, y in enumerate(y_pos):
                    positions.append( [x,y] )
                    print positions[-1]
        
        for position in positions:
            position = np.array(positions)
            
    odor_dataset = od.Dataset()
            
    key = 0
    for position in positions:
        odor_trace = make_false_odor_trace(gm, timestamps, position)
        odor_dataset.odor_traces.setdefault(key, odor_trace)
        key += 1
    
    return odor_dataset 
    
    
    
def fit_1d_gaussian(odor_dataset, t, axis=0, keys=None, plot=True, lims=[-1,1], ignore_parameter_names=[]):
    
    if keys is None:
        keys = odor_dataset.odor_traces.keys()
    
    ordinate = []
    odor = []
    odor_std = []
    for key in keys:
        odor_trace = odor_dataset.odor_traces[key]
        ordinate.append( odor_trace.position[axis] )
        index_at_t = np.argmin( np.abs(odor_trace.timestamps - t) )
        odor.append( odor_trace.voltage[index_at_t] )
        odor_std.append( odor_trace.voltage_std[index_at_t] )
        
    ordinate = np.array(ordinate)
    odor = np.array(odor)   
    odor_std = np.array(odor_std)   
        
    print ordinate
    print odor
        
    # now fit gaussian to data
    gm = data_fit.models.GaussianModel1D()
    inputs = [ordinate]
    
    #return odor, ordinate
    
    gm.fit_with_guess(odor, inputs, ignore_parameter_names=ignore_parameter_names)
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(ordinate, odor, 'ob')
        for i, pt in enumerate(ordinate):
            ax.vlines(pt, odor[i]-odor_std[i], odor[i]+odor_std[i], linewidth=2)
        
        x = np.arange(lims[0], lims[1], 0.001)
        vals = gm.get_val(x)
        ax.plot(x, vals)
        
        fpl.adjust_spines(ax, ['left', 'bottom'])
        
        ax.set_xlabel('x position, m')
        ax.set_ylabel('odor value, ethanol')
        ax.set_title('mean and std dev of measured odor values and gaussian fit')
        
        
    
    return gm, ordinate, odor, odor_std
    
    
def fit_1d_gaussian_time_varying(odor_dataset, tmin=15.2, tmax=18, tres=0.1, num=None, colormap='jet', tres_for_plot=0.5, axis=0, keys=None, lims=None, ignore_parameter_names=[], plot=True):
    
    if lims is None:
        if axis==0:
            lims = [-.3,1]
        elif axis==1:
            lims = [-.1,.1]

    if keys is None:
        sets, keys_to_position = od.find_sets(odor_dataset,axis)
        lengths_of_sets = np.asarray([len(s) for s in sets.values()])
        set_to_use = np.argmax(lengths_of_sets)
        if num is not None:
            set_to_use = num
        keys = sets[sets.keys()[set_to_use]]
    
    timestamps = np.arange(tmin, tmax, tres)
    ordinate = []
    odor = []
    odor_std = []
    for t in timestamps:
        odor_data_at_time_t = []
        odor_std_data_at_time_t = []
        for key in keys:
            odor_trace = odor_dataset.odor_traces[key]
            
            odor_at_time_t = np.interp(t, odor_trace.timestamps, odor_trace.voltage)
            odor_data_at_time_t.append(odor_at_time_t)
            
            odor_std_at_time_t = np.interp(t, odor_trace.timestamps, odor_trace.voltage_std)
            odor_std_data_at_time_t.append(odor_std_at_time_t)
            
            if t == timestamps[0]:
                ordinate.append( odor_trace.position[axis] )
        
        odor.append(np.array(odor_data_at_time_t))
        odor_std.append(np.array(odor_std_data_at_time_t))
        
    ordinate = np.array(ordinate)
        
        
    # now fit gaussian to data
    gm = data_fit.models.GaussianModel1D_TimeVarying()
    inputs = [ordinate]
    gm.fit(timestamps, odor, inputs)
    
    if plot:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        norm = matplotlib.colors.Normalize(tmin, tmax)
        cmap = matplotlib.cm.ScalarMappable(norm, colormap)
        t_arr = np.arange(tmin, tmax, tres_for_plot)
        
        for t in t_arr:
        
            # find index of timestamps array that corresponds to this t
            index = np.argmin(np.abs(timestamps-t))
        
            color = cmap.to_rgba(t)

            for i, pt in enumerate(ordinate):
                ax.vlines(pt, odor[index][i]-odor_std[index][i], odor[index][i]+odor_std[index][i], color='black', linewidth=1)
            
            ax.plot(ordinate, odor[index], 'o', color=color, markersize=8)
            x = np.arange(lims[0], lims[1], 0.001)
            vals = gm.get_val([t, [x]])
            ax.plot(x, vals, color=color, linewidth=2)
            
        fpl.adjust_spines(ax, ['left', 'bottom'])
        
        ax.set_xlabel('x position, m')
        ax.set_ylabel('odor value, ethanol')
        ax.set_title('mean and std dev of measured odor values and time varying gaussian fit\ncolor=time')
    
    return gm
    
    
    
    
    
    
    
    
# fitting routine

# sets, keys_to_positions = od.find_sets(odor_dataset, axis=0, min_keys_in_set=4, threshold=0.004)
    
    
#################################################################################################
# 2D Gaussian stuff - doesn't work with the data I've collected
#################################################################################################

    
def fit_2d_gaussian(odor_dataset, t=13.6, keys=None):
    #od.calc_windspeed(odor_dataset, position=[0, 0.16])
    
    if keys is None:
        keys = odor_dataset.odor_traces.keys()
    
    # guess for center:
    #x0_guess = t*odor_dataset.windspeed + 0        
    x0_guess = 0
    y0_guess = -0.01
        
    x = []
    y = []
    odor = []
    for key in keys:
        odor_trace = odor_dataset.odor_traces[key]
        x.append( odor_trace.position[0] )
        y.append( odor_trace.position[1] )
        index_at_t = np.argmin( np.abs(odor_trace.timestamps - t) )
        odor.append( odor_trace.voltage[index_at_t] )
        
    x = np.array(x)
    y = np.array(y)
    odor = np.array(odor)   
        
    print x
    print y
    print odor
        
    # now fit gaussian to data
    gm = data_fit.models.GaussianModel2D()
    inputs = [x,y]
    gm.fit_with_guess(odor, inputs)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    gm.show_fit(odor, inputs, ax=ax, lims=[[-.2,1], [-.1,.1]], resolution=0.001)
    ax.set_xlabel('x position, m')
    ax.set_ylabel('y position, m')
    ax.set_title('Odor heatmap (gaussian fit), one point in time')
    
    
    return gm
    
    
def fit_2d_gaussian_moving(odor_dataset):
    
    t_list = np.arange(0, 2, .1)
    
    gm_list = []
    for t in t_list:
        gm = fit_2d_gaussian(odor_dataset, t=t, plot=False)
        gm_list.append(gm)
        
    mean_0_list = np.zeros_like(t_list)
    std_0_list = np.zeros_like(t_list)
    mean_1_list = np.zeros_like(t_list)
    std_1_list = np.zeros_like(t_list)
    magnitude_list = np.zeros_like(t_list)
    for i, gm in enumerate(gm_list):
        mean_0_list[i] = gm.parameters['mean_0']
        std_0_list[i] = gm.parameters['std_0']
        mean_1_list[i] = gm.parameters['mean_1']
        std_1_list[i] = gm.parameters['std_1']
        magnitude_list[i] = gm.parameters['magnitude']
        
    parameter_list = [mean_0_list, std_0_list, mean_1_list, std_1_list, magnitude_list]
    lm_list = []
    
    
    for i, param in enumerate(parameter_list):
        lm = data_fit.models.LinearModel(parameters={'slope': 1, 'intercept': 0})
        print parameter_list[i]
        lm.fit(parameter_list[i], t_list)
        print lm.parameters
        print
        lm_list.append(copy.copy(lm))
    
        
        
    return gm_list, parameter_list, lm_list
    
    
    
def fit_2d_gaussian_moving_builtin(odor_dataset):
    
    timestamps = np.arange(13, 16, .01)
    
    def get_positions(odor_dataset):
        x = []
        y = []
        for key, odor_trace in odor_dataset.odor_traces.items():
            x.append( odor_trace.position[0] )
            y.append( odor_trace.position[1] )
        return [np.array(x), np.array(y)]        
    
    def get_odor(timestamps, odor_dataset):
        odor = []
        for t in timestamps:
            odor_at_t = []
            for key, odor_trace in odor_dataset.odor_traces.items():
                index_at_t = np.argmin( np.abs(odor_trace.timestamps - t) )
                odor_at_t.append( odor_trace.voltage[index_at_t] )
            odor.append(np.array(odor_at_t))
        return odor
        
    positions = get_positions(odor_dataset)
    odor = get_odor(timestamps, odor_dataset)
            
    gm = data_fit.models.GaussianModel2D_TimeVarying()
    gm.fit(timestamps, odor, positions)
    
    return gm    
    
