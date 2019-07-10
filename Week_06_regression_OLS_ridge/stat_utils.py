# Utility functions

import numpy as np


def column_corr(A, B, dof=0):
    """Efficiently compute correlations between columns of two matrices
    
    Does NOT compute full correlation matrix btw `A` and `B`; returns a 
    vector of correlation coefficients. FKA ccMatrix."""
    zs = lambda x: (x-np.nanmean(x, axis=0))/np.nanstd(x, axis=0, ddof=dof)
    rTmp = np.nansum(zs(A)*zs(B), axis=0)
    n = A.shape[0]
    # make sure not to count nans
    nNaN = np.sum(np.logical_or(np.isnan(zs(A)), np.isnan(zs(B))), 0)
    n = n - nNaN
    r = rTmp/n
    return r