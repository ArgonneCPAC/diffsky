""" """

import numpy as np
from jax import random as jran

from .. import tw_utils


def _mae(pred, target):
    diff = pred - target
    return np.mean(np.abs(diff))


def test_tw_interp_kern():
    """Enforce triweight interpolation roughly agrees with linear interpolation"""
    ran_key = jran.key(0)
    x_min, x_max = 0, 10
    y_min, y_max = -1, 1
    xarr = np.linspace(x_min, x_max, 2_000)

    n_tests = 100
    for __ in range(n_tests):
        ran_key, x_key, y_key = jran.split(ran_key, 3)
        x_table = np.sort(jran.uniform(x_key, minval=x_min, maxval=x_max, shape=(3,)))
        y_table = jran.uniform(y_key, minval=y_min, maxval=y_max, shape=(3,))
        y_interp = np.interp(xarr, x_table, y_table)
        y_tw_interp = tw_utils._tw_interp_kern(xarr, *x_table, *y_table)
        loss = _mae(y_tw_interp, y_interp)
        assert loss < 0.1
