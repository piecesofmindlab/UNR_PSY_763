"""Utilities for fMRI section of CogNeuro Connector for Data8"""

import numpy as np
import nibabel
import scipy.stats
import warnings
import numpy as np


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
