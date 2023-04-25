"""
"""
from jax import jit as jjit
from jax import random as jran
from dsps.experimental.diffburst import DEFAULT_DBURST


@jjit
def _mc_burst(ran_key, gal_logsm, gal_logssfr, params):
    n = gal_logsm.shape[0]
    fburst_key, dburst_key = jran.split(ran_key, 2)
    fburst = jran.uniform(fburst_key, minval=0, maxval=0.1, shape=(n,))
    dburst = jran.uniform(
        dburst_key, minval=DEFAULT_DBURST, maxval=DEFAULT_DBURST + 0.1, shape=(n,)
    )
    return fburst, dburst
