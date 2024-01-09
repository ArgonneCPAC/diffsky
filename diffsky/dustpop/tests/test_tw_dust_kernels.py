"""
"""
import numpy as np

from .. import tw_dust_kernels as tdk


def test_something():
    n = 1_000
    wave_micron = np.logspace(-5, 5, n)
    av = 1.0
    delta = -0.4
    funo = 0.1
    res = tdk.triweight_k_lambda(wave_micron, av, delta, funo)
    assert res.shape == (n,)
    assert np.all(np.isfinite(res))
    assert np.all(res >= 0)
