"""
"""
import numpy as np
from ..utils import _sigmoid, _inverse_sigmoid


def test_sigmoid_inversion():
    xarr = np.linspace(-10, 10, 500)

    x0, k, ylo, yhi = 0, 0.1, -5, 5
    y = _sigmoid(xarr, x0, k, ylo, yhi)
    x2 = _inverse_sigmoid(y, x0, k, ylo, yhi)
    assert np.allclose(xarr, x2, rtol=1e-4)
