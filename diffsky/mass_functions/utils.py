""" """

import numpy as np

from ..utils import _sig_slope, _sigmoid  # noqa


def get_1d_arrays(*args):
    """Return a list of ndarrays of the same length.

    Each arg must be either an ndarray of shape (npts, ), or a scalar.

    """
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)

    result = [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
    return result
