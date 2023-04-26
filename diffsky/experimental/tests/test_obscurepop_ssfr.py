"""
"""
import numpy as np
from jax import random as jran
from ..obscurepop_ssfr import mc_funobs, BPOP_DEFAULT_U_PARAMS


def test_mc_funobs():
    ran_key = jran.PRNGKey(0)
    n_gals = 500
    logsm = np.random.uniform(8, 12, n_gals)
    logssfr = np.random.uniform(-12, -8, n_gals)

    funobs = mc_funobs(ran_key, logsm, logssfr, BPOP_DEFAULT_U_PARAMS)
    assert np.all(np.isfinite(funobs))
