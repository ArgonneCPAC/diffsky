"""
"""
import numpy as np
from jax import random as jran
from ..obscurepop import mc_funobs, BPOP_DEFAULT_PARMS


def test_mc_funobs():
    ran_key = jran.PRNGKey(0)
    n_gals = 500
    logsm = np.random.uniform(8, 12, n_gals)
    logssfr = np.random.uniform(-12, -8, n_gals)

    mc_funobs(ran_key, logsm, logssfr, BPOP_DEFAULT_PARMS)
