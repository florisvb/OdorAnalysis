import help_functions as hf
import fly_plot_lib.plot as fpl

threshold_odor = 10
odor = True

data_in_odor = []
data_not_in_odor = []
odor_stimulus_list = ['none', 'on']

for odor_stimulus in odor_stimulus_list:
    keys = opa.get_keys_with_odor_before_post(config, dataset, threshold_odor=threshold_odor, odor_stimulus=odor_stimulus, threshold_distance_min=100, odor=odor)
    
    if 0:        
        keys = []
        for key in keys_tmp:
            trajec = dataset.trajecs[key]
            add_key = True
            if trajec.positions[0,0] < 0.3:
                add_key = False
            if odor:
                frames_in_odor = np.where(trajec.odor > threshold_odor)[0]
                if len(frames_in_odor) < 40:
                    add_key = False
            if add_key:
                keys.append(key)

    mean_accels_in_odor = []
    mean_accels_not_in_odor = []
    for key in keys:
        trajec = dataset.trajecs[key]
        frames_in_odor = np.where(trajec.odor > threshold_odor)[0]
        frames_not_in_odor = np.where(trajec.odor < threshold_odor)[0]
        '''
        
        odor_blocks = hf.find_continuous_blocks(frames_in_odor, 5, return_longest_only=False)
            
        
        for i, block in enumerate(odor_blocks):
            if len(block) < 1:
                continue
            frame0 = block[0]
            frame0 = np.max([frame0, 0])
            frame1 = block[-1]
            frames = np.arange(frame0, frame1)
            
            
            #mean = (np.diff(trajec.speed[frames])).tolist()
            mean = trajec.speed[frames].tolist()
            mean_accels.extend(mean)
        '''
        
        mean_accels_in_odor.extend((np.diff(trajec.speed[frames_in_odor])*trajec.fps).tolist())
        mean_accels_not_in_odor.extend((np.diff(trajec.speed[frames_not_in_odor])*trajec.fps).tolist())
                     
    data_in_odor.append(np.array(mean_accels_in_odor))
    data_not_in_odor.append(np.array(mean_accels_not_in_odor))
    
    print odor_stimulus, ': ', np.mean(mean_accels), ' +/- ', np.std(mean_accels)

# in odor
fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
nbins = 75
bins = np.linspace(-.05,.05,nbins)*100
fpl.histogram(ax, data_in_odor, bins=bins, bin_width_ratio=0.8, colors=['black', 'red'], edgecolor='none', normed=True, show_smoothed=True, bar_alpha=1, curve_line_alpha=0)
ax.set_xlim(bins[0], bins[-1])
fpl.adjust_spines(ax, ['left', 'bottom'])
ax.set_xlabel('Acceleration, m/s2')
ax.set_ylabel('Occurences, normalized')

save_figure_path = os.path.join(config.path, config.figure_path, 'activity/')
figname = save_figure_path + 'accceleration_histogram_in_odor' + '.pdf'
fig.savefig(figname, format='pdf')

# not in odor
fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
nbins = 75
bins = np.linspace(-.05,.05,nbins)*100
fpl.histogram(ax, data_not_in_odor, bins=bins, bin_width_ratio=0.8, colors=['black', 'blue'], edgecolor='none', normed=True, show_smoothed=True, bar_alpha=1, curve_line_alpha=0)
ax.set_xlim(bins[0], bins[-1])
fpl.adjust_spines(ax, ['left', 'bottom'])
ax.set_xlabel('Acceleration, m/s2')
ax.set_ylabel('Occurences, normalized')    
        
save_figure_path = os.path.join(config.path, config.figure_path, 'activity/')
figname = save_figure_path + 'accceleration_histogram_not_in_odor' + '.pdf'
fig.savefig(figname, format='pdf')
        
        
        
