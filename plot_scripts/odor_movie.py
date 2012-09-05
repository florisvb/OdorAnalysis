import sys, os
from optparse import OptionParser

import flydra_analysis_tools as fat
fad = fat.flydra_analysis_dataset
tac = fat.trajectory_analysis_core

import numpy
np = numpy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

import fly_plot_lib.plot as fpl

sys.path.append('../analysis_modules')
import odor_packet_analysis as opa

import fly_plot_lib
params = {  'figure.figsize': (8,5), 
            'figure.subplot.left': 0.1,
            'figure.subplot.right': 0.95,
            'figure.subplot.bottom': 0.1,
            'figure.subplot.top': 0.95,
          }

fly_plot_lib.set_params.pdf(params)

def play_odor_movie(path, config, trajec, axis='xy', axis_slice=0.04, save=False):
    
    # animation parameters
    anim_params = {'t': 10, 'xlim': [-.2,1], 'ylim': [-0.15,.15], 't_max': 15., 'dt': 0.05, 'resolution': 0.001, 'ghost_tail': 20, 'ghost_start': 0, 'trajec': trajec}
    
    # load gaussian model
    gm = opa.get_gaussian_model(path, config)

    # prep plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # post and such
    if axis == 'xy':
        axes=[0,1,2]
        post = patches.Circle(config.post_center[0:2], config.post_radius, color='black', zorder=10)
    if axis == 'xz':
        axes=[0,2,1]
        post = patches.Rectangle([-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black', zorder=10)
    if axis == 'yz':
        axes=[1,2,0]
        post = patches.Rectangle([-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black', zorder=10)
    
    # initialize image
    odor_image, extent = gm.get_array_2d_slice(anim_params['t'], anim_params['xlim'], anim_params['ylim'], anim_params['resolution'], axis=axes[-1], axis_slice=axis_slice)
    
    norm = matplotlib.colors.Normalize(0, 100)
    color_mappable = matplotlib.cm.ScalarMappable(norm, plt.get_cmap('jet'))
    im = ax.imshow( odor_image, extent=extent, origin='lower', cmap=plt.get_cmap('jet'), norm=norm, alpha=0.5, zorder=0)
    
    # post
    ax.add_artist(post)
    
    # plot trajectory
    indices = np.arange(anim_params['ghost_start'], anim_params['ghost_start']+anim_params['ghost_tail'],1)
    #fly, = ax.plot(trajec.positions[indices][:,axes[0]], trajec.positions[indices][:,axes[1]], '.')
    x = trajec.positions[:,axes[0]]
    y = trajec.positions[:,axes[1]]
    color = trajec.odor
    orientation = trajec.heading_smooth
    flies = fpl.get_wedges_for_heading_plot(x, y, color, orientation, size_radius=0.015, size_angle=20, colormap='jet', colornorm=None, edgecolors='none', alpha=1, flip=True, deg=False, nskip=0)
    ax.add_collection(flies)
    
    def init_plot(): 
        #fly.set_data([],[])
        flies.set_color('none')
        return im, flies
    
    def updatefig(*args):
        anim_params['ghost_start'] += 1
        if anim_params['ghost_tail'] + anim_params['ghost_start'] > anim_params['trajec'].length:
            anim_params['ghost_start'] = 0
        indices = np.arange(anim_params['ghost_start'], anim_params['ghost_start']+anim_params['ghost_tail'],1).tolist()
                                
        odor_image, extent = gm.get_array_2d_slice(trajec.time_relative_to_last_pulse[indices[-1]], anim_params['xlim'], anim_params['ylim'], anim_params['resolution'], axis=axes[-1], axis_slice=axis_slice)
        
        colors = ['none' for i in range(anim_params['trajec'].length)]
        for i in indices:
            colors[i] = color_mappable.to_rgba(anim_params['trajec'].odor[i])
        
        flies.set_facecolors(colors)
        flies.set_edgecolors('none')
        
        im.set_array(odor_image)
        
        if save:
            print indices[0]
            save_movie_path = config.movie_tmp_path
            frame_prefix = '_tmp'
            frame_prefix = os.path.join(path, save_movie_path, frame_prefix)
            strnum = str(anim_params['ghost_start'])
            while len(strnum) < 5:
                strnum = '0' + strnum
            frame_name = frame_prefix + '_' + strnum + '_' + '.png'
            fig.savefig(frame_name, format='png')
        
        return im,flies

    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=[-.2, 0, .2, .4, .6, .8, 1], yticks=[-.15, 0, .15])
    ax.set_xlabel('x position, m')
    ax.set_ylabel('y position, m')
    ani = animation.FuncAnimation(fig, updatefig, init_func=init_plot, fargs=anim_params, interval=50, blit=True)
    
    plt.show()
    
######################################################################################################################

def play_odor_movie_2_axes(path, config, trajec, axis_slice_xy=0.04, axis_slice_xz=0.01, save=False, nskip=0):
    
    # animation parameters
    anim_params = {'t': 10, 'xlim': [-.2,1], 'ylim': [-0.15,.15], 'zlim': [-.15, .15], 't_max': 15., 'dt': 0.05, 'resolution': 0.001, 'ghost_tail': 20, 'ghost_start': 0, 'trajec': trajec}
    
    # load gaussian model
    gm = opa.get_gaussian_model(path, config)

    # prep plot
    fig = plt.figure()
    ax_xy = fig.add_subplot(211)
    ax_xz = fig.add_subplot(212)
    
    # post and such
    height = config.post_center[2]-config.ticks['z'][0]
    post_xy = patches.Circle(config.post_center[0:2], config.post_radius, color='black', zorder=10)
    post_xz = patches.Rectangle([-1*config.post_radius, config.ticks['z'][0]], config.post_radius*2, height, color='black', zorder=10)
    
    # initialize image
    odor_image_xy, extent_xy = gm.get_array_2d_slice(anim_params['t'], anim_params['xlim'], anim_params['ylim'], anim_params['resolution'], axis=2, axis_slice=axis_slice_xy)
    odor_image_xz, extent_xz = gm.get_array_2d_slice(anim_params['t'], anim_params['xlim'], anim_params['zlim'], anim_params['resolution'], axis=1, axis_slice=axis_slice_xz)
    
    norm = matplotlib.colors.Normalize(0, 100)
    color_mappable = matplotlib.cm.ScalarMappable(norm, plt.get_cmap('jet'))
    im_xy = ax_xy.imshow( odor_image_xy, extent=extent_xy, origin='lower', cmap=plt.get_cmap('jet'), norm=norm, alpha=0.5, zorder=0)
    im_xz = ax_xz.imshow( odor_image_xz, extent=extent_xz, origin='lower', cmap=plt.get_cmap('jet'), norm=norm, alpha=0.5, zorder=0)
    
    # post
    ax_xy.add_artist(post_xy)
    ax_xz.add_artist(post_xz)
    
    # plot trajectory
    x = trajec.positions[:,0]
    y = trajec.positions[:,1]
    z = trajec.positions[:,2]
    color = trajec.odor
    orientation_xy = trajec.heading_smooth
    tac.calc_heading_for_axes(trajec, axis='xz')
    orientation_xz = trajec.heading_smooth_xz
    flies_xy = fpl.get_wedges_for_heading_plot(x, y, color, orientation_xy, size_radius=0.015, size_angle=20, colormap='jet', colornorm=None, edgecolors='none', alpha=1, flip=True, deg=False, nskip=0)
    flies_xz = fpl.get_wedges_for_heading_plot(x, z, color, orientation_xz, size_radius=0.015, size_angle=20, colormap='jet', colornorm=None, edgecolors='none', alpha=1, flip=True, deg=False, nskip=0)
    ax_xy.add_collection(flies_xy)
    ax_xz.add_collection(flies_xz)
    
    def init_plot(): 
        flies_xy.set_color('none')
        flies_xz.set_color('none')
        return im_xy, im_xz, flies_xy, flies_xz
    
    def updatefig(*args):
        anim_params['ghost_start'] += 1 + nskip
        if anim_params['ghost_tail'] + anim_params['ghost_start'] > anim_params['trajec'].length:
            anim_params['ghost_start'] = 0
        indices = np.arange(anim_params['ghost_start'], anim_params['ghost_start']+anim_params['ghost_tail'],1).tolist()
                                
        
        odor_image_xy, extent_xy = gm.get_array_2d_slice(trajec.time_relative_to_last_pulse[indices[-1]], anim_params['xlim'], anim_params['ylim'], anim_params['resolution'], axis=2, axis_slice=axis_slice_xy)
        odor_image_xz, extent_xz = gm.get_array_2d_slice(trajec.time_relative_to_last_pulse[indices[-1]], anim_params['xlim'], anim_params['zlim'], anim_params['resolution'], axis=1, axis_slice=axis_slice_xz)
        
        colors = ['none' for i in range(anim_params['trajec'].length)]
        for i in indices:
            colors[i] = color_mappable.to_rgba(anim_params['trajec'].odor[i])
        
        flies_xy.set_facecolors(colors)
        flies_xy.set_edgecolors('none')
        
        flies_xz.set_facecolors(colors)
        flies_xz.set_edgecolors('none')
        
        im_xy.set_array(odor_image_xy)
        im_xz.set_array(odor_image_xz)
        
        if save:
            print indices[0]
            save_movie_path = config.movie_tmp_path
            frame_prefix = '_tmp'
            frame_prefix = os.path.join(path, save_movie_path, frame_prefix)
            strnum = str(anim_params['ghost_start'])
            while len(strnum) < 5:
                strnum = '0' + strnum
            frame_name = frame_prefix + '_' + strnum + '_' + '.png'
            fig.savefig(frame_name, format='png')
        
        return im_xy, im_xz, flies_xy, flies_xz

    fpl.adjust_spines(ax_xy, ['left'], yticks=[-.15, 0, .15])
    fpl.adjust_spines(ax_xz, ['left', 'bottom'], xticks=[-.2, 0, .2, .4, .6, .8, 1], yticks=[-.15, 0, .15])
    ax_xz.set_xlabel('x position, m')
    ax_xz.set_ylabel('z position, m')
    ax_xy.set_ylabel('y position, m')
    ani = animation.FuncAnimation(fig, updatefig, init_func=init_plot, fargs=anim_params, interval=50, blit=True)
    
    plt.show()
    
    
    
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--path", type="str", dest="path", default='',
                        help="path to empty data folder, where you have a configuration file")
    parser.add_option("--key", type="str", dest="key", default='',
                        help="key of trajec to make a movie of")        
    parser.add_option("--axis", type="str", dest="axis", default='xy',
                        help="axis")  
    parser.add_option("--nskip", type="int", dest="nskip", default=0,
                        help="n frames to skip (to speed it up)")  
    parser.add_option("--save", dest="save", action="store_true", default=False,
                        help="save movie?")         
                        
    (options, args) = parser.parse_args()
    
    path = options.path    
    key = options.key
    axis = options.axis
    save = options.save
    nskip = options.nskip
    
    sys.path.append(path)
    import analysis_configuration
    config = analysis_configuration.Config()
    
    culled_dataset_name = os.path.join(path, config.culled_datasets_path, config.culled_dataset_name)
    culled_dataset = fad.load(culled_dataset_name)
    
    trajec = culled_dataset.trajecs[key]
    
    play_odor_movie_2_axes(path, config, trajec, save=save, nskip=nskip)
    
    
    
    #fps (frames per second) controls the play speed
    #mencoder 'mf://*.png' -mf type=png:fps=30 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o animation.avi
    
    
    # 0_10257
    
    
