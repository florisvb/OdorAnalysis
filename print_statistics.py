import flydra_analysis_tools as fat
fad = fat.flydra_analysis_dataset
import numpy as np
import odor_packet_analysis as opa
import time

def time_struct_to_float_time(t):
    return t.tm_hour + t.tm_min/60. + t.tm_sec/3600.

def print_landing_takeoff_timings(config, dataset, print_stats=True):
    if print_stats:
        print 'Landings: '
    landing_times = []
    landers = fad.get_keys_with_attr(dataset, ['post_behavior'], ['landing'])
    for lander in landers:
        trajec = dataset.trajecs[lander]
        t = time.localtime(trajec.timestamp_epoch + trajec.time_fly[-1])
        landing_times.append(time_struct_to_float_time(t))
        if print_stats:
            print lander, time_struct_to_float_time(t)
    if print_stats:
        print
        print 'Takeoffs: '
    takeoff_times = []
    takeoffs = fad.get_keys_with_attr(dataset, ['post_behavior'], ['takeoff'])
    for takeoff in takeoffs:
        trajec = dataset.trajecs[takeoff]
        t = time.localtime(trajec.timestamp_epoch + trajec.time_fly[-1])
        takeoff_times.append(time_struct_to_float_time(t))
        if print_stats:
            print takeoff, time_struct_to_float_time(t)
    
    landing_times = np.asarray(landing_times)
    takeoff_times = np.asarray(takeoff_times)
    
    if print_stats:
        print
        print 'Residency Time: '
    residency_times = []
    for landing_time in landing_times:
        possible_residency_times = takeoff_times - landing_time
        if np.max(possible_residency_times) > 0:
            positive_residency_times = np.where(possible_residency_times > 0)[0]
            if print_stats:
                print np.min(possible_residency_times[positive_residency_times])
            
    return np.asarray(residency_times)


def print_stats(config, dataset):

    threshold_distance_min = 0.1
    threshold_odor = 10
    
    for odor_stimulus in config.odor_stimulus.keys():
        print
        print 'Experiment Conditions: Odor ', odor_stimulus.replace('_', ' ').title()
        print
        print 'Idependent of Odor: '
        print
        nlanding = len(fad.get_keys_with_attr(dataset, ['post_behavior','odor_stimulus'], ['landing',odor_stimulus]))
        nboomerang = len(fad.get_keys_with_attr(dataset, ['post_behavior','odor_stimulus'], ['boomerang',odor_stimulus]))
        ntakeoff = len(fad.get_keys_with_attr(dataset, ['post_behavior','odor_stimulus'], ['takeoff',odor_stimulus]))
        keys = fad.get_keys_with_attr(dataset, ['odor_stimulus'], [odor_stimulus])
        n_flies = len(keys)
        
        mean_speeds = []
        frame_length = []
        for key in keys:
            trajec = dataset.trajecs[key]
            mean_speeds.append( np.mean(trajec.speed) )
            frame_length.append( len(trajec.speed) )
        
        mean_speeds = np.array(mean_speeds)
        frame_length = np.array(frame_length)
        
        print 'num flies: ', n_flies
        print 'num landing: ', nlanding
        print 'landing ratio: ', nlanding / float(n_flies)
        print 'num boomerang: ', nboomerang
        print 'boomerang ratio: ', nboomerang / float(n_flies)
        print 'num takeoff: ', ntakeoff
        print 'mean speed: ', np.mean(mean_speeds), ' +/- ', np.std(mean_speeds)
        print 'mean trajec length (secs): ', np.mean(frame_length) / (100.), ' +/- ', np.std(frame_length) / (100.)
        
        #############################################################################################################################################################
        print
        print 'Experienced Odor: '
        print
        keys = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=threshold_odor, odor_stimulus=odor_stimulus)
        
        nlanding = len(fad.get_keys_with_attr(dataset, ['post_behavior','odor_stimulus'], ['landing',odor_stimulus], keys=keys))
        nboomerang = len(fad.get_keys_with_attr(dataset, ['post_behavior','odor_stimulus'], ['boomerang',odor_stimulus], keys=keys))
        ntakeoff = len(fad.get_keys_with_attr(dataset, ['post_behavior','odor_stimulus'], ['takeoff',odor_stimulus], keys=keys))
        keys_with_attr = fad.get_keys_with_attr(dataset, ['odor_stimulus'], [odor_stimulus], keys=keys)
        n_flies = len(keys_with_attr)
        
        mean_speeds = []
        frame_length = []
        mean_speeds_in_odor = []
        mean_speeds_not_in_odor = []
        mean_speeds_not_in_odor_after = []
        residency_time = []
        n_saccades_close_to_post = []
        frames_until_saccade = []
        heading_after_saccade = []
        
        for key in keys_with_attr:
            trajec = dataset.trajecs[key]
            mean_speeds.append( np.mean(trajec.speed) )
            frame_length.append( len(trajec.speed) )
            
            frames_in_odor = np.where(trajec.odor > threshold_odor)[0].tolist()
            frames_in_odor_and_center = []
            for f in frames_in_odor:
                if trajec.distance_to_post[f] > 0.05:
                    if np.linalg.norm(trajec.positions[f,1:]) < 0.07:
                        frames_in_odor_and_center.append(f)
                            
            frames_not_in_odor = np.where(trajec.odor < 1)[0].tolist()
            frames_not_in_odor_prior_to_odor = []
            for f in frames_not_in_odor:
                if f < frames_in_odor[0]:
                    if trajec.distance_to_post[f] > 0.05:
                        if np.linalg.norm(trajec.positions[f,1:]) < 0.07:
                            frames_not_in_odor_prior_to_odor.append(f)
                    
            frames_not_in_odor_after_to_odor = []
            for f in frames_not_in_odor:
                if f > frames_in_odor[-1]:
                    if trajec.distance_to_post[f] > 0.05:
                        if np.linalg.norm(trajec.positions[f,1:]) < 0.07:
                            frames_not_in_odor_after_to_odor.append(f)
                
            if len(frames_in_odor_and_center) > 0:
                mean_speeds_in_odor.append(np.mean(trajec.speed[frames_in_odor_and_center]))
            
            if len(frames_not_in_odor_prior_to_odor) > 0:
                mean_speeds_not_in_odor.append(np.mean(trajec.speed[frames_not_in_odor_prior_to_odor]))
            
            if len(frames_not_in_odor_after_to_odor) > 0:
                mean_speeds_not_in_odor_after.append(np.mean(trajec.speed[frames_not_in_odor_after_to_odor]))
                
            if trajec.residency_time is not None:
                residency_time.append(trajec.residency_time)
                
            ## saccade analysis
            nsacs = 0
            for sac in trajec.saccades:
                if np.min(trajec.distance_to_post[sac]) < 0.05:
                    nsacs += 1
            n_saccades_close_to_post.append(nsacs)
            
            ## after odor, how long till next saccade?
            frames_in_odor = np.where(trajec.odor > threshold_odor)[0].tolist()
            for sac in trajec.saccades:
                if sac[0] > frames_in_odor[-1]:
                    frames_until_saccade.append(sac[0] - frames_in_odor[-1])
                    heading_after_saccade.append(np.abs(trajec.heading_smooth[sac[-1]]))
                    break
        
        mean_speeds = np.array(mean_speeds)
        frame_length = np.array(frame_length)
        
        print 'num flies: ', n_flies
        print 'num landing: ', nlanding
        print 'landing ratio: ', nlanding / float(n_flies)
        print 'num boomerang: ', nboomerang
        print 'boomerang ratio: ', nboomerang / float(n_flies)
        print 'num takeoff: ', ntakeoff
        print 'mean speed: ', np.mean(mean_speeds), ' +/- ', np.std(mean_speeds)
        print 'mean speed while in odor: ', np.mean(mean_speeds_in_odor), ' +/- ', np.std(mean_speeds_in_odor)
        print 'mean speed while NOT in odor PRIOR: ', np.mean(mean_speeds_not_in_odor), ' +/- ', np.std(mean_speeds_not_in_odor)
        print 'mean speed while NOT in odor AFTER: ', np.mean(mean_speeds_not_in_odor_after), ' +/- ', np.std(mean_speeds_not_in_odor_after)
        print 'mean trajec length (secs): ', np.mean(frame_length) / (100.), ' +/- ', np.std(frame_length) / (100.)
        print 'mean residency time (sec): ', np.mean(residency_time) / (100.), ' +/- ', np.std(residency_time) / (100.), len(residency_time)
        print 'n saccades near post: ', np.mean(n_saccades_close_to_post), ' +/- ', np.std(n_saccades_close_to_post)
        print 'n sec till saccade after odor: ', np.mean(frames_until_saccade)/100., ' +/- ', np.std(frames_until_saccade)/100.
        print 'heading after saccade: ', np.mean(heading_after_saccade)/100., ' +/- ', np.std(heading_after_saccade)/100.
        print
        #############################################################################################################################################################
        print
        print 'Experienced NO Odor: '
        print
        keys = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=threshold_odor, odor_stimulus=odor_stimulus, odor=False)
        
        nlanding = len(fad.get_keys_with_attr(dataset, ['post_behavior','odor_stimulus'], ['landing',odor_stimulus], keys=keys))
        nboomerang = len(fad.get_keys_with_attr(dataset, ['post_behavior','odor_stimulus'], ['boomerang',odor_stimulus], keys=keys))
        ntakeoff = len(fad.get_keys_with_attr(dataset, ['post_behavior','odor_stimulus'], ['takeoff',odor_stimulus], keys=keys))
        keys_with_attr = fad.get_keys_with_attr(dataset, ['odor_stimulus'], [odor_stimulus], keys=keys)
        n_flies = len(keys_with_attr)
        
        mean_speeds = []
        frame_length = []
        mean_speeds_near_center = []
        residency_time = []
        n_saccades_close_to_post = []
        for key in keys_with_attr:
            trajec = dataset.trajecs[key]
            mean_speeds.append( np.mean(trajec.speed) )
            frame_length.append( len(trajec.speed) )
            
            frames = range(0,trajec.length)
            frames_to_use = []
            for f in frames:
                if trajec.distance_to_post[f] > 0.05:
                    if np.linalg.norm(trajec.positions[f,1:]) < 0.07:
                        frames_to_use.append(f)
            
            if len(frames_to_use) > 0:
                mean_speeds_near_center.append(np.mean(trajec.speed[frames_to_use]))
                
            if trajec.residency_time is not None:
                residency_time.append(trajec.residency_time)
                
            nsacs = 0
            for sac in trajec.saccades:
                if np.min(trajec.distance_to_post[sac]) < 0.05:
                    nsacs += 1
            n_saccades_close_to_post.append(nsacs)
        
        mean_speeds = np.array(mean_speeds)
        frame_length = np.array(frame_length)
        
        print 'num flies: ', n_flies
        print 'num landing: ', nlanding
        print 'landing ratio: ', nlanding / float(n_flies)
        print 'num boomerang: ', nboomerang
        print 'boomerang ratio: ', nboomerang / float(n_flies)
        print 'num takeoff: ', ntakeoff
        print 'mean speed: ', np.mean(mean_speeds), ' +/- ', np.std(mean_speeds)
        print 'mean speed while near center: ', np.mean(mean_speeds_near_center), ' +/- ', np.std(mean_speeds_near_center)
        print 'mean trajec length (secs): ', np.mean(frame_length) / (100.), ' +/- ', np.std(frame_length) / (100.)
        print 'mean residency time (sec): ', np.mean(residency_time) / (100.), ' +/- ', np.std(residency_time) / (100.), len(residency_time)
        print 'n saccades near post: ', np.mean(n_saccades_close_to_post), ' +/- ', np.std(n_saccades_close_to_post)
        print
        print '**********************************'
        print
        
        
        
        
        
        
        
        
        
def count_post_behaviors(dataset, post_behavior, keys=None):
    
    if keys is None:
        keys = dataset.trajecs.keys()
        
    n = 0
    for key in keys:
        trajec = dataset.trajecs[key]
        for behavior in trajec.post_behavior:
            if behavior == post_behavior:
                n += 1
    
    return n
    
    
        
        
        
        
        
        
        
        
