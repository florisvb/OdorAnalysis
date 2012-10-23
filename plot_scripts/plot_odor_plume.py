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
import xyz_axis_grid

def plot_odor_plume(config, gm):

    fig = plt.figure()
    ax_xy, ax_xz, ax_yz = xyz_axis_grid.get_axes(fig)
    
    # get static plume
    static_time = 12.
    gm_static = gm.get_gaussian_model_at_time_t(static_time)
    
    # set x axis to be huge
    gm_static.parameters['mean_0'] = 0
    gm_static.parameters['std_0'] = 10000000000000000
    
    #xlim, ylim, resolution, axis=2, axis_slice=0)
    odor_image_xy, extent_xy = gm_static.get_array_2d_slice([-.2,1], [-.15,.15], .001, axis=2, axis_slice=.04)
    odor_image_xz, extent_xz = gm_static.get_array_2d_slice([-.2,1], [-.15,.15], .001, axis=1, axis_slice=.01)
    odor_image_yz, extent_yz = gm_static.get_array_2d_slice([-.15,.15], [-.15,.15], .001, axis=0, axis_slice=0)
    
    ax_xy.imshow(odor_image_xy, extent=extent_xy, origin='lower')
    ax_xz.imshow(odor_image_xz, extent=extent_xz, origin='lower')
    ax_yz.imshow(odor_image_yz, extent=extent_yz, origin='lower')
    
    xyz_axis_grid.set_spines_and_labels(ax_xy, ax_xz, ax_yz)
    
    figure_path = os.path.join(config.path, config.figure_path)
    save_figure_path = os.path.join(figure_path, 'odor_traces/')
    pdf_name_with_path = os.path.join(save_figure_path, 'odor_plume_model.pdf')


    fig.savefig(pdf_name_with_path, format='pdf')    
    
    
    
    
    
    
    
    
    
    
    
