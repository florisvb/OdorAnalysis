#!/usr/bin/env python
import roslib; roslib.load_manifest('ros_flydra')
import rospy
from ros_flydra.srv import *
import time
import os
import pickle

"""
Python serial interface to the IO Rodeo solid state relay expansion board for 
the Arduino Nano. 

Author: Will Dickson, IO Rodeo Inc.

Copyright 2010  IO Rodeo Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import serial
import numpy as np
import matplotlib.pyplot as plt

import odor_dataset as od
import floris_plot_lib as fpl

class BasicSSR(serial.Serial):

    def __init__(self,**kwargs):
        super(BasicSSR,self).__init__(**kwargs)
           
    def pulse(self, ssr_num, pulse_length, pulse_interval, exp_length, record_data, units='sec'):
        if units == 'sec':
            pulse_length = int(pulse_length*1000)
            pulse_interval = int(pulse_interval*1000)
            exp_length = int(exp_length*1000)
        self.write('[%d, %d, %d, %d, %d]\n'%(ssr_num, pulse_length, pulse_interval, exp_length, record_data))
        
    def listen(self):
        raw_data = []
        do_work = 1
        
        print 'recording data!'
        while do_work:
            t = str(time.time())
            data = self.readline()
            #print data
            if 'done' in data:
                do_work = 0
            else:
                data = t + ',' + data
                raw_data.append(data)
        print 'done recording data!'
        return raw_data
        
    def listen_for_control_signal_only(self):
        time_start = int(time.time())
    
        fname = time.strftime("odor_control_signal_%Y%m%d_%H%M%S",time.localtime())
        f = open(fname, 'w')
        
        s = 'time_start,' + str(time_start) + '\n'
        f.write(s)
        
        do_work = 1
        print 'recording data!'
        while do_work:
            t = str(time.time() - time_start)
            data = self.readline()
            if len(data) > 0:            
                if 'done' in data:
                    do_work = 0
                    f.close()
                else:
                    #t = time.time()-time_start
                    s = data[0:-2] + ',' + t + '\n'
                    f.write(s)
                    print 'my data: ', data
        print 'done recording data!'
        
        
###################################################################
# Processing data
###################################################################
        
def process_raw_data(raw_data):
    print 'parsing'
        
    time_computer = np.zeros(len(raw_data))
    time_arduino = np.zeros(len(raw_data))
    ssr_val = np.zeros(len(raw_data))
    voltage = np.zeros(len(raw_data))
        
    for i, data in enumerate(raw_data):
        try:
            data_split = data.split(',')
            time_computer[i] = data_split[0]
            time_arduino[i] = data_split[2]
            voltage[i] = data_split[3]
            ssr_val[i] = data_split[1]
        except:
            print 'this data packet not working'
            
    return time_computer, time_arduino, ssr_val, voltage
    
    
def save_as_odor_trace(time_computer, time_arduino, ssr_val, voltage, resolution=0.001):
    
    odor_trace = od.Odor_Trace()
    
    time_computer_interp = np.arange(time_computer[0], time_computer[-1], resolution)
    voltage_interp = np.interp(time_computer_interp, time_computer, voltage)
    ssr_val_interp = np.ceil(np.interp(time_computer_interp, time_computer, ssr_val))
    
    odor_trace.voltage = voltage_interp
    odor_trace.timestamps = time_computer_interp
    odor_trace.signal = ssr_val_interp
    
    return odor_trace
    
def remove_steady_state_from_trace(odor_trace, index_0, index_1):
    mean_steady_state = np.mean(odor_trace.voltage[index_0:index_1])
    odor_trace.voltage[0:odor_trace.voltage_start+1] = mean_steady_state
    odor_trace.voltage -= mean_steady_state
    
def remove_steady_state_from_dataset(odor_dataset, index_0, index_1):
    for key, odor_trace in odor_dataset.odor_traces.items():
        remove_steady_state_from_trace(odor_trace, index_0, index_1)
        
def calc_mean_odor_trace(odor_dataset, ignore_traces=[1,2,3,4], led_position_vector=[0,0,0]):
    mean_odor_trace = od.Odor_Trace()
    
    positions_led = None
    traces = None
    timestamps = None
    signal = None
    
    ignore_traces_strings = ['_' + str(i) for i in ignore_traces]
    
    for key, odor_trace in odor_dataset.odor_traces.items():
        if key[-2:] in ignore_traces_strings:
            continue
        if timestamps is None:
            timestamps = odor_trace.timestamps
            
        if positions_led is None:
            positions_led = odor_trace.position_led
        else:
            positions_led = np.vstack((positions_led, odor_trace.position_led))
            
        if traces is None:
            traces = odor_trace.voltage
        else:
            traces = np.vstack((traces, odor_trace.voltage))
            
        if signal is None:
            signal = odor_trace.signal
        else:
            signal = np.vstack((signal, odor_trace.signal))
            
    mean_odor_trace.signal = np.mean(signal, axis=0)            
    
    
    # find when odor signal turned on:
    index_when_signal_turned_on = np.argmax(np.diff(mean_odor_trace.signal))
    mean_odor_trace.timestamps = timestamps - timestamps[index_when_signal_turned_on]
    
    mean_odor_trace.position_led = np.mean(positions_led, axis=0)
    mean_odor_trace.voltage = np.mean(traces, axis=0)
    mean_odor_trace.position = mean_odor_trace.position_led - np.asarray(led_position_vector)
    mean_odor_trace.voltage_std = np.std(traces, axis=0)
    odor_dataset.mean_odor_trace = mean_odor_trace
    
    
def plot_mean_odor_trace(odor_dataset, ignore_traces=[1,2,3,4]):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ignore_traces_strings = ['_' + str(i) for i in ignore_traces]
    for key, odor_trace in odor_dataset.odor_traces.items():
        if key[-2:] in ignore_traces_strings:
            continue
        color = 'gray'
        ax.plot(odor_dataset.mean_odor_trace.timestamps, odor_trace.signal, color)
        ax.plot(odor_dataset.mean_odor_trace.timestamps, odor_trace.voltage, color)
        
    ax.plot(odor_dataset.mean_odor_trace.timestamps, odor_dataset.mean_odor_trace.signal, 'red')
    ax.plot(odor_dataset.mean_odor_trace.timestamps, odor_dataset.mean_odor_trace.voltage, 'red')
        
    fpl.adjust_spines(ax, ['left', 'bottom'])
    ax.set_xlabel('time, sec')
    ax.set_ylabel('control signal and odor response')
    ax.set_title('raw odor traces + signal, and mean (red)')
    
    
def plot_mean_odor_trace_from_file(filename):
    
    f = open(filename)
    odor_dataset = pickle.load(f)
    calc_mean_odor_trace(odor_dataset)  
    plot_mean_odor_trace(odor_dataset)
    
    
def find_delay(odor_trace, units='time'):
    signal = odor_trace.signal - np.mean(odor_trace.signal)
    signal /= np.std(signal)
    voltage = odor_trace.voltage - np.mean(odor_trace.voltage)
    voltage /= np.std(voltage)
    #delay_in_indices = np.argmax(np.correlate(signal, voltage[::-1], "same"))
    delay_in_indices = np.abs( signal.shape[0] - np.argmax(np.correlate(signal, voltage, "full")) )
    delay_in_seconds = delay_in_indices*np.mean(np.diff(odor_trace.timestamps))
    if units == 'time':
        return delay_in_seconds
    elif units == 'indices':
        return delay_in_indices
        
def calc_signal_onset(odor_trace):
    odor_trace.signal_onsets = np.where(np.diff(odor_trace.signal) == 1)[0].astype(int)
        
def split_pulse_train(odor_trace, odor_dataset=None, basekey='p1_'):

    if odor_dataset is None:
        odor_dataset = od.Dataset()
    
    delay_in_indices = find_delay(odor_trace, units='indices')
    calc_signal_onset(odor_trace)
    interval_in_indices = int( np.mean(np.diff(odor_trace.signal_onsets)) )
    dt = np.mean(np.diff(odor_trace.timestamps))

    index_shift = 4000
    n = 0
    for onset in odor_trace.signal_onsets:
        n += 1
        start_index_signal = onset - index_shift
        stop_index_signal = onset + interval_in_indices - index_shift

        start_index_voltage = onset - index_shift + delay_in_indices
        stop_index_voltage = onset + interval_in_indices - index_shift + delay_in_indices

        total_indices = stop_index_voltage - start_index_signal
        print stop_index_voltage, start_index_signal, total_indices

        new_odor_trace = od.Odor_Trace()
        new_odor_trace.position = odor_trace.position
        try:
            new_odor_trace.position_led = odor_trace.position_led
        except:
            pass
        new_odor_trace.timestamps = np.copy(odor_trace.timestamps[start_index_signal:stop_index_voltage])
        new_odor_trace.timestamps -= new_odor_trace.timestamps[0]

        new_odor_trace.voltage = np.zeros(total_indices)
        new_odor_trace.signal = np.zeros(total_indices)

        new_odor_trace.signal[0:interval_in_indices] = odor_trace.signal[start_index_signal:stop_index_signal]
        new_odor_trace.voltage[delay_in_indices:delay_in_indices+interval_in_indices] = odor_trace.voltage[start_index_voltage:stop_index_voltage]
        
        new_odor_trace.voltage_start = delay_in_indices

        new_key = basekey + str(n)
        odor_dataset.odor_traces.setdefault(new_key, new_odor_trace)
        
    return odor_dataset
    
def plot_odor_traces(odor_dataset):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for key, odor_trace in odor_dataset.odor_traces.items():
        ax.plot(odor_trace.timestamps, odor_trace.signal)
        ax.plot(odor_trace.timestamps, odor_trace.voltage)
        
    ax.set_xlabel('time, sec')
    ax.set_ylabel('control signal and odor response')
    
def get_odor_dataset_from_means(path, led_position_vector=[0,0,0]):
    
    new_odor_dataset = od.Dataset()
    
    cmd = 'ls ' + path
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    try: all_filelist.remove('')
    except: pass

    for filename in all_filelist:
        filename_with_path = os.path.join(path, filename)
        f = open(filename_with_path)
        odor_dataset = pickle.load(f)
        f.close()
        calc_mean_odor_trace(odor_dataset, led_position_vector=led_position_vector)  
        key = filename
        new_odor_dataset.odor_traces.setdefault(key, odor_dataset.mean_odor_trace)
        
    return new_odor_dataset
        
        
###################################################################
# Run Experiments
###################################################################

def run_experiment(savepath='', dev=None, pulse_length=0.4, pulse_interval=10, exp_length=200, record_data=1):
    if dev is None:
        dev = BasicSSR(port='/dev/ttyUSB0',timeout=1, baudrate=115200)
    time.sleep(2.0) # Sleep for serial reset of arduino
    
    ssr_num = 0
    dev.pulse(ssr_num, pulse_length, pulse_interval, exp_length, record_data)
    time.sleep(0.5)
    raw_data = dev.listen()
    time_computer, time_arduino, ssr_val, voltage = process_raw_data(raw_data)
    
    odor_trace = save_as_odor_trace(time_computer, time_arduino, ssr_val, voltage, resolution=0.001)
    
    # get position from flydra
    fsl = Flydra_Service_Listener()
    odor_trace.position_led = fsl.get_mean_led_position()
    
    basekey = time.strftime("%Y%m%d_%H%M%S_",time.localtime())
    odor_dataset = split_pulse_train(odor_trace, odor_dataset=None, basekey=basekey)
    remove_steady_state_from_dataset(odor_dataset, -1000, -1)
    
    odor_dataset.raw_odor_trace = odor_trace
    
    fname = time.strftime("odor_dataset_%Y%m%d_%H%M%S",time.localtime())
    fname_with_path = os.path.join(savepath, fname)
    f = open(fname_with_path, 'w')
    pickle.dump(odor_dataset, f)
    f.close()
    
    return odor_trace, odor_dataset
    
def run_fly_experiment(savepath='', dev=None, pulse_length=0.4, pulse_interval=30, exp_length=54000, record_data=0):
    if dev is None:
        dev = BasicSSR(port='/dev/ttyUSB0',timeout=1, baudrate=115200)
    time.sleep(2.0) # Sleep for serial reset of arduino
    
    ssr_num = 0
    dev.pulse(ssr_num, pulse_length, pulse_interval, exp_length, record_data)
    time.sleep(0.5)
    raw_data = dev.listen_for_control_signal_only()
    

def run_diverse_fly_experiment():

    savepath=''
    dev=None
    ssr_num = 0
    record_data=0

    if dev is None:
        dev = BasicSSR(port='/dev/ttyUSB0',timeout=1, baudrate=115200)
    time.sleep(2.0) # Sleep for serial reset of arduino
    
    localtime = (time.localtime()).tm_hour + (time.localtime()).tm_min / 60. + (time.localtime()).tm_sec / 360.
    while localtime > 12 and localtime < 24:
        time.sleep(2)
        localtime = (time.localtime()).tm_hour + (time.localtime()).tm_min / 60. + (time.localtime()).tm_sec / 360.
    
    hours_to_keep_odor_on = 3
    pulse_length = hours_to_keep_odor_on*60*60
    pulse_interval = 0
    exp_length = pulse_length + 30
    dev.pulse(ssr_num, pulse_length, pulse_interval, exp_length, record_data)
    time.sleep(0.5)
    raw_data = dev.listen_for_control_signal_only()
    
    hours_to_pulse_odor = 6
    pulse_length = 0.4
    pulse_interval = 30
    exp_length = hours_to_pulse_odor*60*60
    dev.pulse(ssr_num, pulse_length, pulse_interval, exp_length, record_data)
    time.sleep(0.5)
    raw_data = dev.listen_for_control_signal_only()
            
    
###################################################################
# Flydra stuff
###################################################################

class Flydra_Service_Listener:
    def __init__(self):
        rospy.wait_for_service("flydra_super_packet_service")
        self.get_latest_flydra_data = rospy.ServiceProxy("flydra_super_packet_service", super_packet_service)

    def get_position_from_flydra_data(self):
        superpacket = self.get_latest_flydra_data().packets
        for packet in superpacket.packets:
            if len(packet.objects) == 1:
                for obj in packet.objects:
                    position = [obj.position.x, obj.position.y, obj.position.z]
                    return position             

    def get_mean_led_position(self, n_avg=20):
        positions = None
        n = 0
        while n<n_avg:
            position = self.get_position_from_flydra_data()
            if positions is None:
                positions = np.array(position)
            else:
                positions = np.vstack((positions, position))
            n+=1
        positions_avg = np.mean(positions, axis=0)
        return positions_avg
    
            
    

'''    

d_mm = 3.15
d = d_mm/1000.
r = d/2.
area = np.pi*r**2
wind_speed = .4
flow_rate = area*wind_speed
flow_rate_sccs = flow_rate*100**3
flow_rate_sccm = flow_rate_sccs * 60.

yield: 187 sccm
'''








if __name__ == '__main__':
    #run_experiment(filename='odor_dataset', pulse_length=100, pulse_interval=1000, num_pulses=0, exp_length=1*60*1000, record_data=1, odor_type='acetone', resistance=100, num_trials=1, gain=10)
    
    run_odor_experiment()
    
    print 'done'







