# Make stimuli for pRF mapping
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.transform import resize

"""
Each of the functions here creates a different type of pRF mapping stimulus (or does some other
useful thing for making arrays used in pRF analysis)
"""

PIXEL_RESOLUTION = 101 # Default pixel resolution for all functions

def _force_list(x, convert_tuple=False):
    """Assure that a variable is a list. 

    Any variable that is not a list (including a tuple) is enclosed in a list."""
    if x is None:
        return []
    if convert_tuple and isinstance(x, tuple):
        x = list(x)
    if not isinstance(x, list):
        x = [x]
    return x


def _circ_dist(a, b, mx=2*np.pi):
    """Calculate distance in circular coordinates

    Parameters
    ----------
    a, b are scalars / arrays
    mx is max of circular cooordinate system (min is presumed to be 0 for now)
    """
    m = np.minimum(np.abs(a-b), mx-np.abs(a-b))
    return m

def make_hrf(shape='twogamma', tr=1, pttp=5, nttp=15, pnr=6, ons=0, pdsp=1, ndsp=1, s=None):
    """Create canonical hemodynamic response filter

    Parameters
    ----------
    shape : 
        HRF general shape {'twogamma' [, 'boynton']}
    tr : 
        HRF sample frequency (default: 1s/16, OK: [1e-3 .. 5])
    pttp : 
        time to positive (response) peak (default: 5 secs)
    nttp : 
        time to negative (undershoot) peak (default: 15 secs)
    pnr : 
        pos-to-neg ratio (default: 6, OK: [1 .. Inf])
    ons : 
        onset of the HRF (default: 0 secs, OK: [-5 .. 5])
    pdsp : 
        dispersion of positive gamma PDF (default: 1)
    ndsp : 
        dispersion of negative gamma PDF (default: 1)
    s : 
        sampling range (default: [0, ons + 2 * (nttp + 1)])
    
    Returns
    -------
    h : HRF function given within [0 .. onset + 2*nttp]
    s : HRF sample points
    
    Notes
    -----
    The pttp and nttp parameters are increased by 1 before given
    as parameters into the scipy.stats.gamma.pdf function (which is a property
    of the gamma PDF!)

    Converted to python from BVQXtools 
    Version:  v0.7f
    Build:    8110521
    Date:     Nov-05 2008, 9:00 PM CET
    Author:   Jochen Weber, SCAN Unit, Columbia University, NYC, NY, USA
    URL/Info: http://wiki.brainvoyager.com/BVQXtools
    """

    # Input checks
    if tr > 5:
        tr = 1/16
    elif tr < 0.001:
        tr = 0.001
    if not shape.lower() in ('twogamma', 'boynton'):
        warnings.warn('Shape can only be "twogamma" or "boynton"')
        shape = 'twogamma'
    if s is None:
        s = np.arange(0,(ons + 2 * (nttp + 1)), tr) - ons
    else:
        s = np.arange(np.min(s),np.max(s),tr) - ons;

    # computation (according to shape)
    h = np.zeros((len(s),));
    if shape.lower()=='boynton':
        # boynton (single-gamma) HRF
        h = scipy.stats.gamma.pdf(s, pttp + 1, pdsp)
    elif shape.lower()=='twogamma':
        gpos = scipy.stats.gamma.pdf(s, pttp + 1, pdsp)
        gneg = scipy.stats.gamma.pdf(s, nttp + 1, ndsp) / pnr
        h =  gpos-gneg             
    # normalize for convolution
    h /= np.sum(h)
    return s,h

def make_wedge_im(theta, wedge_width=np.pi/2, size_degrees=10, pixel_resolution=PIXEL_RESOLUTION):
    """Get a wedge image at a given resolution and size_degrees

    Parameters
    ----------
    theta : scalar
        angle of wedge in radians
    wedge_width : scalar
        polar angle width of wedge in radians
    pixel_resolution : scalar 
        pixels per degree
    size_degrees : scalar
        radius of whole wedge stimulus (whole screen) in visual degrees
        assumes square screen.
    """
    t = np.linspace(-size_degrees, size_degrees, pixel_resolution)
    xg, yg = np.meshgrid(t, t)
    # Convert to polar coords
    r = np.sqrt(xg**2 + yg**2)
    th = np.arctan2(xg, yg) + np.pi
    # Flip theta in x direction
    th = th[:, ::-1]    
    wedge_bool = (_circ_dist(th, theta) < wedge_width/2) & (r <= size_degrees)
    return wedge_bool.astype('int')

def make_ring_im(ring_indices, ring_radii='glab', pixel_resolution=PIXEL_RESOLUTION, size_degrees=10):
    """Create a ring stimulus 

    Creates an indexed image with a number of (usually log-spaced) concentric rings,
    each with a different image. The first variable `ring_indices` (a list or tuple) selects 
    which of these indices (and thus which concentric rings) are "on" for this image.

    Parameters
    ----------
    ring_indices : list | tuple
        indices for which of the concentric rings are "on"
    ring_radii : str or list
        two special strings can be specified: 'glab' or 'log<n>' (e.g. log8) 
        'glab' specifies the ring stimulus parameters as used by the Gallant lab
        for retinotopic mapping
        'log<n>' specifies <n> log-spaced bins (starting at 1)
        if a list is given, the list defines the radii for the edges of the 
        concentric circles.
    pixel_resolution : scalar
        pixels in the stimulus (square stimulus is assumed)
    size_degrees : scalar
        radius of the whole display, in visual degrees
    """
    t = np.linspace(-size_degrees, size_degrees, pixel_resolution)
    xg, yg = np.meshgrid(t, t)
    # Convert to polar coords
    r = np.sqrt(xg**2 + yg**2)
    if ring_radii=='glab':
        # Derived from stimulus used in Glab retinotopic mapping.
        ring_wid_fixed = [9, 12, 19, 26, 33, 42, 53, 63]
        ring_edge_pos = np.cumsum(ring_wid_fixed).astype('float')
        ring_edge_pos_pct = ring_edge_pos / np.sum(ring_wid_fixed)
        ring_edges = np.hstack([0, ring_edge_pos_pct]) * size_degrees
    elif ring_radii.startswith('log'):
        n_rings = int(ring_radii[3:])
        ring_edges = np.hstack([0, np.logspace(np.log10(1), np.log10(size_degrees), n_rings)])
    else:
        ring_edges = ring_radii
    # indexed image for radii
    ri = np.zeros(r.shape)
    for ii, (re1, re2) in enumerate(zip(ring_edges[:-1], ring_edges[1:]), 1):
        bi = (r>=re1) & (r <=re2)
        ri[bi] = ii
    ring_bool = (ri==ring_indices[0]+1) | (ri==ring_indices[1]+1) & (r <= size_degrees)
    return ring_bool.astype('int')

def make_wedge_sequence(start_pos, direction='cw', n_trs=360, n_cycles=15, **kwargs):
    """
    direction : str
        'cw' (clockwise) or 'ccw' (counter clockwise)
    """
    tr_per_cycle = n_trs / n_cycles
    rad_per_tr = 2*np.pi / tr_per_cycle
    x = 1 if direction=='cw' else -1
    theta_sequence = np.mod(np.arange(start_pos, x*rad_per_tr*n_trs + start_pos, x*rad_per_tr), np.pi*2)
    return np.array([make_wedge_im(theta, **kwargs) for theta in theta_sequence])

def make_ring_sequence(start_indices, max_radius=10, direction='exp', n_trs=360, n_cycles=15, n_ring_positions=8, **kwargs):
    """
    direction : str
        'cw' (clockwise) or 'ccw' (counter clockwise)
    """
    s1, s2 = start_indices
    tr_per_cycle = n_trs / float(n_cycles)
    tr_per_ring = tr_per_cycle / n_ring_positions
    if direction=='exp':
        #x = 1
        seq_tmp = np.arange(n_trs)
    elif direction=='contr':
        #x = -1
        seq_tmp = np.arange(n_trs, 0, -1)
    seq_tmp = np.floor(np.mod(seq_tmp, tr_per_cycle)/tr_per_ring)
    aa = np.mod(seq_tmp+s1, n_ring_positions)
    bb = np.mod(seq_tmp+s2, n_ring_positions)
    radius_sequence = np.array(zip(aa, bb))
    rings = np.array([make_ring_im(radius, **kwargs) for radius in radius_sequence])
    return rings

def make_bar_im(width, orientation, phase, size_degrees=10, pixel_resolution=PIXEL_RESOLUTION):
    """Make a mask for a bar passing across the visual field
    
    Parameters
    ----------
    width : scalar
        bar width in degrees of visual angle
    orientation : scalar
        orientation of bar in radians(0 = vertical, 90=horizontal, rotates clockwise)
    phase : float, [0-1]
        how far across the visual field the bar is.
    size_degrees : scalar
        radius of area for bar stimuli in degrees
    """
    t = np.linspace(-size_degrees, size_degrees, pixel_resolution)
    xg, yg = np.meshgrid(t, t)
    r = np.sqrt(xg**2 + yg**2)
    xg *= np.sin(orientation)
    yg *= np.cos(orientation)
    full_area = r < size_degrees
    p1 = size_degrees*2 * phase + -size_degrees - width/2.0
    p2 = size_degrees*2 * phase + -size_degrees + width/2.0
    bar = ((xg+yg) >= p1) & ((xg+yg) <= p2)
    mask = bar & full_area
    return mask.astype(np.float)

def make_bar_sequence(bar_width = 3, size_degrees=10, pixel_resolution=PIXEL_RESOLUTION):
    """Make sequence of bar masks for full experiment run"""
    bar_seq = []
    for orid in [90, 325, 180, 45, 270, 135, 0, 225]: #np.arange(0, 2*pi, pi/4):
        ori = np.deg2rad(orid)
        for ph in np.linspace(0, 1, 16):
            if (orid not in [0, 90, 180, 270]) and (ph > 0.5):
                # Leave a gap
                bb = np.zeros((pixel_resolution, pixel_resolution))
            else:
                bb = make_bar_im(bar_width, ori, ph, 
                       size_degrees=12.5, pixel_resolution=pixel_resolution)
            bar_seq.append(bb)
    return np.dstack(bar_seq)

def compute_contrast(ims, ctype='pixel_gradient', im_size=(101,101)):
    """Compute contrast for stimulus images
    TODO: implement the following ctypes:
    (a) simple gradient magnitude (easy to compute, prob wrong thing)
    (b) motion energy contrast (gradient in 3d?)
    (c) compute orientation contrast w/ fourier spectrum (bandpass power per image)
    (d) do smarter thing, computing orientation contrast by scale
    
    """
    if ctype=='pixel_gradient':
        xg = np.gradient(ims, axis=0)
        yg = np.gradient(ims, axis=1)
        # Average contrast in each color... Kinda dumb
        mag = ((xg**2+yg**2)**0.5).mean(2)
    else:
        raise NotImplementedError('contrast type not recognized!')
    if ims.shape[:2] != im_size:
        new_im = np.zeros(im_size + (mag.shape[-1],), dtype=mag.dtype)
        for ii in range(mag.shape[-1]):
            new_im[:,:,ii] = resize(mag[...,ii], im_size)
        return new_im
    else:
        return mag
    

def make_X(direction, wedge_start=np.pi/4, ring_start=None, do_z=True, **kwargs):
    """Get design matrix for multiple wedge runs. Convolves with HRF, zscores, concatenates.
    `direction` is a list of strings, e.g. ['cw', 'ccw', 'exp', 'contr']"""
    direction = _force_list(direction, convert_tuple=True)
    t, hrf = make_hrf(tr=1.5)
    seq = []
    for ii, dd in enumerate(direction):
        if dd in ('cw', 'ccw'):
            stim = make_wedge_sequence(wedge_start, direction=dd, **kwargs)
        elif dd in ('exp', 'contr'):
            if dd=='exp' and ring_start is None:
                print("Expanding - first positions = (7, 0)")
                rs = (7, 0)
            elif dd=='contr' and ring_start is None:
                print("Contracting - first positions = (6, 7)")
                rs = (6, 7)
            else: 
                print("ring_start provided!")
                rs = copy.copy(ring_start)
            stim = make_ring_sequence(rs, direction=dd, **kwargs)
        else:
            raise Exception("Bad input! direction must be in ('cw', 'ccw', 'exp', 'contr')")
        n_trs = stim.shape[0]
        stim = np.reshape(stim, [n_trs, -1])
        vc = np.sum(stim, axis=0) > 0
        if ii==0:
            valid_channels = vc
        else:
            valid_channels |= vc
        # HRF
        stim = np.vstack([np.convolve(x, hrf.flatten(), 'full') for x in stim.T]).T[:stim.shape[0]]
        if do_z:
            # Zscore
            stim = zscore(stim, axis=0)
        # Concatenate to list
        seq.append(stim)
    # Concatenate to array
    X = np.vstack(seq)
    # Clip all-zero channels
    X = X[:, valid_channels]
    
    return X, valid_channels

from matplotlib import animation
import matplotlib.gridspec as gridspec

def make_fancy_anim(idx, resp=None, prf=None, stim=None, interval=16*5, normalize_response=True, extent=None):
    """interval appears to be in ms"""
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(4,6))
    gs = gridspec.GridSpec(3,1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1:])
    y = resp[:, idx]
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
    im = ax2.imshow(stim[:,:,0], extent=extent, cmap='gray')
    im2 = ax2.imshow(prf[..., idx], extent=extent, alpha=0.3, cmap='hot')
    ax2.set_yticks([-10, -5, 0, 5, 10])
    ax2.set_ylabel('Visual field\nY position (deg)')
    ax2.set_xlabel('Visual field\nX position (deg)')
    plt.tight_layout()
    # Hide figure, we don't care
    plt.close(fig.number)
    # initialization function: plot the background of each frame
    def init():
        lines.set_data([], [])
        im.set_array(np.zeros((101,101)))
        return (im, lines)
    # animation function. This is called sequentially
    def animate(i):
        lines.set_data(t_plot[:i], y[:i])
        im.set_array(stim[...,i])
        return (im, lines)
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=stim.shape[-1], interval=interval, blit=True)
    return anim

#anim.save('test.gif', writer='imagemagick', fps=10, dpi=100, )
