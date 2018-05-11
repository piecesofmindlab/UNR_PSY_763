"""Utilities for fMRI section of CogNeuro Connector for Data8"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
import nibabel
import scipy.stats
import warnings
import numpy as np
from matplotlib.colors import Normalize

def hrf(shape='twogamma', tr=1, pttp=5, nttp=15, pos_neg_ratio=6, onset=0, pdsp=1, ndsp=1, t=None):
    """Create canonical hemodynamic response filter
    
    Parameters
    ----------
    shape : string, {'twogamma'|'boynton'}
        HRF general shape {'twogamma' [, 'boynton']}
    tr : scalar
        HRF sample frequency, in seconds (default = 2)
    pttp : scalar
        time to positive (response) peak in seconds (default = 5)
    nttp : scalar
        Time to negative (undershoot) peak in seconds (default = 15)
    pos_neg_ratio : scalar
        Positive-to-negative ratio (default: 6, OK: [1 .. Inf])
    onset : 
        Onset of the HRF (default: 0 secs, OK: [-5 .. 5])
    pdsp : 
        Dispersion of positive gamma PDF (default: 1)
    ndsp : 
        Dispersion of negative gamma PDF (default: 1)
    t : vector | None
        Sampling range (default: [0, onset + 2 * (nttp + 1)])
    
    Returns
    -------
    h : HRF function given within [0 .. onset + 2*nttp]
    t : HRF sample points
    
    Notes
    -----
    The pttp and nttp parameters are increased by 1 before given
    as parameters into the scipy.stats.gamma.pdf function (which is a property
    of the gamma PDF!)

    Based on hrf function in matlab toolbox `BVQXtools`; converted to python and simplified by ML 
    Version:  v0.7f
    Build:    8110521
    Date:     Nov-05 2008, 9:00 PM CET
    Author:   Jochen Weber, SCAN Unit, Columbia University, NYC, NY, USA
    URL/Info: http://wiki.brainvoyager.com/BVQXtools
    """

    # Input checks
    if not shape.lower() in ('twogamma', 'boynton'):
        warnings.warn('Shape can only be "twogamma" or "boynton"')
        shape = 'twogamma'
    if t is None:
        t = np.arange(0, (onset + 2 * (nttp + 1)), tr) - onset
    else:
        t = np.arange(np.min(t), np.max(t), tr) - onset;

    # Create filter
    h = np.zeros((len(t), ))
    if shape.lower()=='boynton':
        # boynton (single-gamma) HRF
        h = scipy.stats.gamma.pdf(t, pttp + 1, pdsp)
    elif shape.lower()=='twogamma':
        gpos = scipy.stats.gamma.pdf(t, pttp + 1, pdsp)
        gneg = scipy.stats.gamma.pdf(t, nttp + 1, ndsp) / pos_neg_ratio
        h =  gpos-gneg 
    h /= np.sum(h)
    return t, h

def hrf_convolve(X, hh):
    """Convolve design matrix with hrf

    Parameters
    ----------
    X : array
        Stimulus design matrix
    hh : hrf (sampled at correct resolution)

    """
    Xh = np.zeros_like(X)
    for i, x in enumerate(X.T):
        Xh[:, i] = np.convolve(x, hh, mode='full')[:len(x)]
    return Xh
    
def get_brain(file):
    """Take the average of some 4D timecourse data to get a brain on which to overlay activity.
    
    Parameters
    ----------
    file : str
        path to file to load. Must be .hdf / .mat (version 7.3) file.
    """
    with h5py.File(file) as hf:
        dd = hf['data'].value
        brain = dd.T.mean(0)
    return brain

def find_squarish_dimensions(n):
    '''Get row, column dimensions for n elememnts

    Returns (nearly) sqrt dimensions for a given number. e.g. for 23, will
    return [5, 5] and for 26 it will return [6, 5]. For creating displays of
    sets of images, mostly. Always sets x greater than y if they are not
    equal.

    Returns
    -------
    x : int
       larger dimension (if not equal)
    y : int
       smaller dimension (if not equal)
    '''
    sq = np.sqrt(n);
    if round(sq)==sq:
        # if this is a whole number - i.e. a perfect square
        x = sq;
        y = sq;
        return x, y
    # One: next larger square
    x = [np.ceil(sq)]
    y = [np.ceil(sq)]
    opt = [x[0]*y[0]];
    # Two: immediately surrounding numbers
    x += [np.ceil(sq)];
    y += [np.floor(sq)];
    opt += [x[1]*y[1]];
    Test = np.array([o-n for o in opt])
    Test[Test<0] = 1000; # make sure negative values will not be chosen as the minimum
    GoodOption = np.argmin(Test);
    x = x[GoodOption]
    y = y[GoodOption]
    return x, y

def overlay_brain(data, brain, threshold=None, 
                  cmap='RdBu_r', vmin=None, vmax=None):
    alpha = (np.abs(data) > threshold).astype(np.float)
    alpha_brain = alpha.reshape(brain.shape)
    nrm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    nrm2 = Normalize()
    br = nrm2(brain)
    cmap = getattr(plt.cm, cmap)
    data_color = cmap(nrm(data))
    data_brain = data_color.reshape(brain.shape + (4,))
    nr, nc = find_squarish_dimensions(brain.shape[0])
    fig, axs = plt.subplots(int(nr), int(nc), figsize=(10,10))
    for i, (ax, dsl, bsl, asl) in enumerate(zip(axs.flatten(), data_brain, brain, alpha_brain)):
        gray_im = plt.cm.gray(nrm2(bsl))
        #print(asl.max())
        im = dsl[...,:3] * asl[:,:,np.newaxis] + gray_im[...,:3] * (1-asl[:,:,np.newaxis])
        ax.imshow(im)
        ax.axis('off')
    for j in range(i+1, len(axs.flatten())):
        axs.flatten()[j].axis('off')