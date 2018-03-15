from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import itertools as itools


def column_corr(A, B, dof=0):
    """Efficiently compute correlations between columns of two matrices
    
    Does NOT compute full correlation matrix btw `A` and `B`; returns a 
    vector of correlation coefficients. """
    zs = lambda x: (x-np.nanmean(x, axis=0))/np.nanstd(x, axis=0, ddof=dof)
    rTmp = np.nansum(zs(A)*zs(B), axis=0)
    n = A.shape[0]
    # make sure not to count nans
    nNaN = np.sum(np.logical_or(np.isnan(zs(A)), np.isnan(zs(B))), 0)
    n = n - nNaN
    r = rTmp/n
    return r

def compute_noise_ceil(data):
    """Computes noise ceiling as mean pairwise correlation between repeats
    
    Parameters
    ----------
    data : array-like
        repeated data; should be (repeats x time x samples [voxels]) 
    
    Returns
    -------
    cc : vector
        correlation per sample (voxel)

    TO DO
    -----
    Make this (optionally) more memory-efficient, with correlations
    computed in chunks
    """
    n_rpts, n_t, n_samples = data.shape
    # Get all pairs of data
    pairs = [p for p in itools.combinations(np.arange(n_rpts), 2)]
    # Preallocate
    r = np.nan*np.zeros((n_samples, len(pairs)))
    for p, (a, b) in enumerate(pairs):
        r[:, p] = column_corr(data[a], data[b])
    cc = np.nanmean(r, 1);
    return cc

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
    sq = np.sqrt(n)
    if round(sq)==sq:
        # if this is a whole number - i.e. a perfect square
        return sq, sq
    # One: next larger square
    x = [np.ceil(sq)]
    y = [np.ceil(sq)]
    opt = [x[0]*y[0]]
    # Two: immediately surrounding numbers
    x += [np.ceil(sq)]
    y += [np.floor(sq)]
    opt += [x[1]*y[1]]
    test = np.array([o-n for o in opt])
    # Make sure negative values will not be chosen as the minimum
    test[test < 0] = np.inf
    idx = np.argmin(test)
    x = x[idx]
    y = y[idx]
    return x, y

def slice_3d_array(volume, axis=2, fig=None, vmin=None, vmax=None, cmap=plt.cm.gray, nr=None, nc=None, 
	figsize=None):
    '''Slices 3D matrix along arbitrary axis

    Parameters
    ----------
    volume : array (3D)
    	data to be sliced
    axis : int | 0, 1, [2] (optional)
      	axis along which to divide the matrix into slices

    Other Parameters
    ----------------
    vmin : float [max(volume)] (optional) 
       color axis minimum
    vmax : float [min(volume)] (optional)
       color axis maximum
    cmap : matplotlib colormap instance [plt.cm.gray] (optional)
    nr : int (optional)
       number of rows
    nc : int (optional)
       number of columns
    '''
    if nr is None or nc is None:
        nc, nr = find_squarish_dimensions(volume.shape[axis])
    if figsize is None:
    	figsize = (10, nr/nc * 10)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if vmin is None:
        vmin = volume.min()
    if vmax is None:
        vmax = volume.max()
    ledges = np.linspace(0, 1, nc+1)[:-1]
    bedges = np.linspace(1, 0, nr+1)[1:]
    width = 1/float(nc)
    height = 1/float(nr)
    bottoms, lefts = zip(*list(itools.product(bedges, ledges)))
    for ni, sl in enumerate(np.split(volume, volume.shape[axis], axis=axis)):
        #ax = fig.add_subplot(nr, nc, ni+1)
        ax = fig.add_axes((lefts[ni], bottoms[ni], width, height))
        ax.imshow(sl.squeeze(), vmin=vmin, vmax=vmax, interpolation="nearest", cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig