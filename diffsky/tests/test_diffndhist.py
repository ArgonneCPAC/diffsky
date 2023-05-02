"""
"""
import numpy as np
from .. import diffndhist


def test_tw_ndhist_returns_correctly_shaped_results():
    npts, ndim = 200, 3
    nbins = 5

    xin = np.zeros((npts, ndim))
    sigin = np.zeros((npts, ndim)) + 0.1
    loin = np.zeros((nbins, ndim))
    hiin = loin + 2.0
    result = diffndhist._tw_ndhist_vmap(xin, sigin, loin, hiin)
    assert result.shape == (nbins,)


def test_tw_ndhist_returns_correct_values_hard_coded_examples():
    """Manually check a few hard-coded specific examples"""
    xc, yc = -0.5, 0.5
    npts, ndim = 200, 2
    nddata = np.tile((xc, yc), npts).reshape((npts, ndim))
    ndsig = np.zeros_like(nddata) + 0.001

    # Choose 3 different cells to compute the histogram
    # Cell 1: (1.0 < x < 2.0) & (0.0 < y < 1.0)
    # Cell 2: (-1.0 < x < 1.0) & (0.0 < y < 3.0)
    # Cell 3: (0.0 < x < 4.0) & (1.0 < y < 5.0)
    nddata_lo = np.array([(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)])
    nddata_hi = np.array([(2.0, 1.0), (1.0, 3.0), (4.0, 5.0)])
    result = diffndhist._tw_ndhist_vmap(nddata, ndsig, nddata_lo, nddata_hi)
    correct_result = np.array((0, npts, 0))
    assert np.allclose(result, correct_result)

    # Choose 3 different cells to compute the histogram
    # Cell 1: (-1.0 < x < 0.0) & (-1.0 < y < 0.0)
    # Cell 2: (0.0 < x < 1.0) & (0.0 < y < 1.0)
    # Cell 3: (1.0 < x < 2.0) & (1.0 < y < 2.0)
    nddata_lo = np.array([(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)])
    nddata_hi = np.array([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    result = diffndhist._tw_ndhist_vmap(nddata, ndsig, nddata_lo, nddata_hi)
    correct_result = np.zeros(3)
    assert np.allclose(result, correct_result)

    # Choose 3 different cells to compute the histogram
    # Cell 1: (-1.0 < x < 0.0) & (0.0 < y < 1.0)
    # Cell 2: (-1.0 < x < 0.0) & (-1.0 < y < 1.0)
    # Cell 3: (1.0 < x < 2.0) & (-2.0 < y < 3.0)
    nddata_lo = np.array([(-1.0, 0.0), (-1.0, -1.0), (1.0, -2.0)])
    nddata_hi = np.array([(0.0, 1.0), (0.0, 1.0), (2.0, 3.0)])
    result = diffndhist._tw_ndhist_vmap(nddata, ndsig, nddata_lo, nddata_hi)
    correct_result = np.array((npts, npts, 0))
    assert np.allclose(result, correct_result)
