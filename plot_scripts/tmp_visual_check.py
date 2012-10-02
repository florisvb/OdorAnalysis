


keys_right = fad.get_keys_with_attr(dataset, 'visual_stimulus', 'right')
keys_left = fad.get_keys_with_attr(dataset, 'visual_stimulus', 'left')

right_times = []
for key in keys_right:
    trajec = dataset.trajecs[key]
    right_times.append(trajec.timestamp_local_float)
right_times = np.array(right_times)

left_times = []
for key in keys_left:
    trajec = dataset.trajecs[key]
    left_times.append(trajec.timestamp_local_float)
left_times = np.array(left_times)

data = [right_times, left_times]

fig = plt.figure()
ax = fig.add_subplot(111)

fpl.histogram(ax, data, bins=50, bin_width_ratio=1, colors=['red', 'green'], edgecolor='none', bar_alpha=1, curve_fill_alpha=0, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False)    

