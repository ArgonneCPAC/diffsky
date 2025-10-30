"""
Test user facing function
"""

import numpy as np
from jax import random as jran

from .. import disk_bulge_sizes as db

LOGM_MIN = 7
LOGM_MAX = 13
Z_MIN = 0
Z_MAX = 5
NTEST = 1_000


def test_disk_median_r50():
    mstar_arr = np.logspace(LOGM_MIN, LOGM_MAX, NTEST)
    z_arr = np.linspace(Z_MIN, Z_MAX, NTEST)
    r50 = db._disk_median_r50(mstar_arr, z_arr)
    assert np.all(np.isfinite(r50))
    assert np.all(r50 >= db.R50_MIN)
    assert np.all(r50 <= db.R50_MAX)


def test_bulge_median_r50():
    mstar_arr = np.logspace(LOGM_MIN, LOGM_MAX, NTEST)
    z_arr = np.linspace(Z_MIN, Z_MAX, NTEST)
    r50 = db._bulge_median_r50(mstar_arr, z_arr)
    assert np.all(np.isfinite(r50))
    assert np.all(r50 >= db.R50_MIN)
    assert np.all(r50 <= db.R50_MAX)


def test_mc_r50_disk_size():
    ran_key = jran.key(0)
    mstar_arr = np.logspace(LOGM_MIN, LOGM_MAX, NTEST)
    z_arr = np.linspace(Z_MIN, Z_MAX, NTEST)
    r50 = db.mc_r50_disk_size(mstar_arr, z_arr, ran_key)
    assert np.all(np.isfinite(r50))
    assert np.all(r50 >= db.R50_MIN)
    assert np.all(r50 <= db.R50_MAX)


def test_mc_r50_bulge_size():
    ran_key = jran.key(0)
    mstar_arr = np.logspace(LOGM_MIN, LOGM_MAX, NTEST)
    z_arr = np.linspace(Z_MIN, Z_MAX, NTEST)
    r50 = db.mc_r50_bulge_size(mstar_arr, z_arr, ran_key)
    assert np.all(np.isfinite(r50))
    assert np.all(r50 >= db.R50_MIN)
    assert np.all(r50 <= db.R50_MAX)
