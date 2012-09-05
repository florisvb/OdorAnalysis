from matplotlib import gridspec
import fly_plot_lib.plot as fpl
import matplotlib.pyplot as plt

def get_axes(fig=None):

    figure_padding = 0.25
    subplot_padding = 0.08

    if fig is None:
        fig = plt.figure(figsize=(8,4.5))
    
    x = 1.2
    y = .3
    z = .3

    aspect_ratio = (y+z+subplot_padding)/(x+y+subplot_padding)

    gs1 = gridspec.GridSpec(2, 2, width_ratios=[x,y])
    gs1.update(left=figure_padding*aspect_ratio, right=1-figure_padding*aspect_ratio, wspace=subplot_padding, hspace=subplot_padding, top=1-figure_padding+subplot_padding, bottom=figure_padding-subplot_padding)
    
    ax_xy = plt.subplot(gs1[0, 0])
    ax_xz = plt.subplot(gs1[1, 0])
    ax_yz = plt.subplot(gs1[1, 1])

    if 1:
        yticks = [-.15, 0, .15]
        xticks = [-.2, 0, 1]
        zticks = [-.15, 0, .15]
    
        ax_xy.set_ylabel('y axis')
        ax_xz.set_ylabel('z axis')
        
        ax_xz.set_xlabel('x axis')
        
        ax_yz.set_xlabel('x axis')
        ax_yz.yaxis.set_label_position('right')
        ax_yz.set_ylabel('z axis')
        
        fpl.adjust_spines(ax_xy, ['left'], yticks=yticks)
        fpl.adjust_spines(ax_xz, ['left', 'bottom'], xticks=xticks, yticks=zticks)
        fpl.adjust_spines(ax_yz, ['right', 'bottom'], xticks=yticks, yticks=zticks)
        
        ax_xy.set_aspect('equal')
        ax_xz.set_aspect('equal')
        ax_yz.set_aspect('equal')
        
    return [ax_xy, ax_xz, ax_yz]
    
    
def set_spines_and_labels(ax_xy, ax_xz, ax_yz):
    
    yticks = [-.15, 0, .15]
    xticks = [-.2, 0, 1]
    zticks = [-.15, 0, .15]
    
    ax_xy.set_xlim(xticks[0], xticks[-1])
    ax_xy.set_ylim(yticks[0], yticks[-1])
    
    ax_xz.set_xlim(xticks[0], xticks[-1])
    ax_xz.set_ylim(zticks[0], zticks[-1])
    
    ax_yz.set_xlim(yticks[0], yticks[-1])
    ax_yz.set_ylim(zticks[0], zticks[-1])
    
    
    fpl.adjust_spines(ax_xy, ['left'], xticks=xticks, yticks=yticks)
    fpl.adjust_spines(ax_xz, ['left', 'bottom'], xticks=xticks, yticks=zticks)
    fpl.adjust_spines(ax_yz, ['right', 'bottom'], xticks=yticks, yticks=zticks)
    
    ax_xy.set_xlabel('')
    ax_xy.set_ylabel('y axis')
    
    ax_xz.set_ylabel('z axis')
    ax_xz.set_xlabel('x axis, upwind negative')
    
    ax_yz.set_xlabel('y axis')
    ax_yz.yaxis.set_label_position('right')
    ax_yz.set_ylabel('z axis')
    
    ax_xy.set_aspect('equal')
    ax_xz.set_aspect('equal')
    ax_yz.set_aspect('equal')
    
    
    
    
    
    
    
    
    
    
    
