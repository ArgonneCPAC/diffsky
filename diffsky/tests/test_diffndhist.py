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


def test_tw_ndhist_weighted_sum_kern():
    ndim = 3

    xin = np.zeros(ndim)
    yin = 1.0
    ndsig = np.zeros(ndim) + 0.1
    ndlo = np.arange(ndim)
    ndhi = ndlo + 1
    res = diffndhist._tw_ndhist_weighted_sum_kern(xin, ndsig, yin, ndlo, ndhi)
    assert res.shape == ()


def test_tw_ndhist_weighted_sum_vmap():
    npts, ndim = 200, 3

    xin = np.zeros((npts, ndim))
    sigin = np.zeros((npts, ndim)) + 0.1
    yin = np.ones(npts)
    ndlo = np.arange(ndim)
    ndhi = ndlo + 1
    result = diffndhist._tw_ndhist_weighted_sum_vmap(xin, sigin, yin, ndlo, ndhi)
    assert result.shape == (npts,)


def test_tw_ndhist_weighted_returns_correctly_shaped_results():
    npts, ndim = 200, 3
    nbins = 5

    xin = np.zeros((npts, ndim))
    sigin = np.zeros((npts, ndim)) + 0.1
    yin = np.ones(npts)
    loin = np.zeros((nbins, ndim))
    hiin = loin + 2.0
    result = diffndhist.tw_ndhist_weighted(xin, sigin, yin, loin, hiin)
    assert result.shape == (nbins,)


def test_tw_ndhist_weighted_returns_correct_values_hard_coded_examples():
    """Manually check a few hard-coded specific examples"""
    xc, yc = -0.5, 0.5
    npts, ndim = 200, 2
    nddata = np.tile((xc, yc), npts).reshape((npts, ndim))
    ndsig = np.zeros_like(nddata) + 0.001
    ydata = np.random.uniform(0, 1, npts)  # randoms for y

    # Choose 3 different cells to compute the histogram
    # Cell 1: (1.0 < x < 2.0) & (0.0 < y < 1.0)
    # Cell 2: (-1.0 < x < 1.0) & (0.0 < y < 3.0)
    # Cell 3: (0.0 < x < 4.0) & (1.0 < y < 5.0)
    nddata_lo = np.array([(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)])
    nddata_hi = np.array([(2.0, 1.0), (1.0, 3.0), (4.0, 5.0)])
    result = diffndhist.tw_ndhist_weighted(nddata, ndsig, ydata, nddata_lo, nddata_hi)
    correct_result = np.array((0, ydata.sum(), 0))
    assert np.allclose(result, correct_result)

    # Choose 3 different cells to compute the histogram
    # Cell 1: (-1.0 < x < 0.0) & (-1.0 < y < 0.0)
    # Cell 2: (0.0 < x < 1.0) & (0.0 < y < 1.0)
    # Cell 3: (1.0 < x < 2.0) & (1.0 < y < 2.0)
    nddata_lo = np.array([(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)])
    nddata_hi = np.array([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    result = diffndhist.tw_ndhist_weighted(nddata, ndsig, ydata, nddata_lo, nddata_hi)
    correct_result = np.zeros(3)
    assert np.allclose(result, correct_result)

    # Choose 3 different cells to compute the histogram
    # Cell 1: (-1.0 < x < 0.0) & (0.0 < y < 1.0)
    # Cell 2: (-1.0 < x < 0.0) & (-1.0 < y < 1.0)
    # Cell 3: (1.0 < x < 2.0) & (-2.0 < y < 3.0)
    nddata_lo = np.array([(-1.0, 0.0), (-1.0, -1.0), (1.0, -2.0)])
    nddata_hi = np.array([(0.0, 1.0), (0.0, 1.0), (2.0, 3.0)])
    result = diffndhist.tw_ndhist_weighted(nddata, ndsig, ydata, nddata_lo, nddata_hi)
    correct_result = np.array((ydata.sum(), ydata.sum(), 0))
    assert np.allclose(result, correct_result)
