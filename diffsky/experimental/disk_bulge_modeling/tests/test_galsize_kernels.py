""""""

import numpy as np
from jax import random as jran

from .. import galsize_kernels as gsk


def test_r50_vs_kern_disk():
    mstar_arr = np.logspace(5, 15, 1_000)
    z = 0.0
    r50 = gsk._r50_disk_kern(mstar_arr, z)
    assert np.all(np.isfinite(r50))
    assert np.all(r50 >= gsk.R50_MIN)
    assert np.all(r50 <= gsk.R50_MAX)


def test_r50_vs_kern_bulge():
    mstar_arr = np.logspace(5, 15, 1_000)
    z = 0.0
    r50 = gsk._r50_bulge_kern(mstar_arr, z)
    assert np.all(np.isfinite(r50))
    assert np.all(r50 >= gsk.R50_MIN)
    assert np.all(r50 <= gsk.R50_MAX)


def test_mc_r50_disk():
    ran_key = jran.key(0)
    mstar_arr = np.logspace(5, 15, 1_000)

    for z in (0.0, 10.0):
        r50 = gsk.mc_r50_disk(mstar_arr, z, ran_key)
        assert np.all(np.isfinite(r50))
        assert np.all(r50 >= gsk.R50_MIN)
        assert np.all(r50 <= gsk.R50_MAX)


def test_mc_r50_bulge():
    ran_key = jran.key(0)
    mstar_arr = np.logspace(5, 15, 1_000)

    for z in (0.0, 10.0):
        r50 = gsk.mc_r50_bulge(mstar_arr, z, ran_key)
        assert np.all(np.isfinite(r50))
        assert np.all(r50 >= gsk.R50_MIN)
        assert np.all(r50 <= gsk.R50_MAX)
