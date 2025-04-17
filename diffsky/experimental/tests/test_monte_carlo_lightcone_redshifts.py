""" """

import numpy as np
from dsps.cosmology import flat_wcdm
from jax import random as jran

from .. import monte_carlo_lightcone_redshifts as mclr


def test_mc_lightcone_redshift():
    ran_key = jran.key(0)
    z_min, z_max = 0.1, 2.0
    npts = 2_000
    cosmo_params = flat_wcdm.PLANCK15
    redshifts = mclr.mc_lightcone_redshift(ran_key, npts, z_min, z_max, cosmo_params)
    assert redshifts.shape == (npts,)
    assert np.all(np.isfinite(redshifts))
    assert np.all(redshifts > z_min)
    assert np.all(redshifts < z_max)
    zbins = np.linspace(z_min, z_max, 10)
    zcounts = np.histogram(redshifts, bins=zbins)[0]
    assert np.all(np.diff(zcounts) > 0)  # always have more galaxies at higher redshift
