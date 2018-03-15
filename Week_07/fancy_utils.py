import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.gridspec as gridspec



def make_fancy_anim(resp=None, gab=None, stim=None, interval=16*5, normalize_response=True, 
    extent=None, cmap=None, alpha=0.3):
    """interval appears to be in ms"""
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(4,6))
    if cmap is None:
        cmap = 'viridis'
    n_pix = 96
    gab_frames = 10
    gs = gridspec.GridSpec(3,1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1:])
    y = resp
    n_trs = len(y)
    t_plot = np.arange(0, n_trs*1.5, 1.5)
    # Normalize 0-1
    if normalize_response:
        y = (y-y.min())/ (y.max()-y.min())
        yl = [-0.2, 1.2]
    else:
        yrange = y.max()-y.min()
        yl = (y.min() - yrange*0.2, y.max() + yrange*0.2)
    # Plot & prettify
    lines, = ax1.plot(t_plot, np.ones(n_trs,))
    ax1.set_ylim(yl)
    ax1.set_ylabel("Predicted\nresponse")
    ax1.set_xlabel("Time (s)")
    # Show image & prettify
    stim_im = ax2.imshow(stim[:,:,0], extent=extent, cmap='gray', vmin=0, vmax=100)
    gab_im = ax2.imshow(gab[..., 0], extent=extent, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)
    ax2.set_yticks([-10, -5, 0, 5, 10])
    ax2.set_ylabel('Visual field\nY position (deg)')
    ax2.set_xlabel('Visual field\nX position (deg)')
    plt.tight_layout()
    # Hide figure, we don't care
    plt.close(fig.number)
    # initialization function: plot the background of each frame
    def init():
        lines.set_data([], [])
        stim_im.set_array(np.zeros((n_pix, n_pix)))
        gab_im.set_array(np.zeros((n_pix, n_pix)))
        return (stim_im, gab_im, lines)
        #return (stim_im, lines)
    # animation function. This is called sequentially
    def animate(i):
        lines.set_data(t_plot[:i], y[:i])
        stim_im.set_array(stim[..., i])
        gab_im.set_array(gab[..., i%gab_frames])
        return (stim_im, gab_im, lines)
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=stim.shape[-1], interval=interval, blit=True)
    return anim

#anim.save('test.gif', writer='imagemagick', fps=10, dpi=100, )
