"""
"""
import numpy as np
from jax import random as jran
from ..obscurepop_burst import mc_funobs, BPOP_DEFAULT_PARAMS


def test_mc_funobs():
    ran_key = jran.PRNGKey(0)
    n_gals = 500
    logsm = np.random.uniform(8, 12, n_gals)
    logfb = np.random.uniform(-4, -1, n_gals)

    funobs = mc_funobs(ran_key, logsm, logfb, BPOP_DEFAULT_PARAMS)
    assert np.all(np.isfinite(funobs))
