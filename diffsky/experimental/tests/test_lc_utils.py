""" """

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from .. import lc_utils as lcu


def test_mc_lightcone_random_redshifts():
    ran_key = jran.key(0)
    npts = 1_000
    z_min, z_max = 0.5, 2.5
    redshift = lcu.mc_lightcone_random_redshifts(
        ran_key, npts, z_min, z_max, DEFAULT_COSMOLOGY
    )
    assert redshift.shape == (npts,)
    assert np.all(np.isfinite(redshift))
    assert np.all(redshift > z_min)
    assert np.all(redshift < z_max)
