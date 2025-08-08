""" """

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from .. import lc_utils as lcu


def test_spherical_shell_comoving_volume():
    z_grid = np.linspace(1, 2, 25)
    vol_shell_grid = lcu.spherical_shell_comoving_volume(z_grid, DEFAULT_COSMOLOGY)
    assert vol_shell_grid.shape == z_grid.shape
    assert np.all(np.isfinite(vol_shell_grid))
    assert np.all(vol_shell_grid > 0)


def test_mc_lightcone_random_redshift():
    ran_key = jran.key(0)
    npts = 1_000
    z_min, z_max = 0.5, 2.5
    redshift = lcu.mc_lightcone_random_redshift(
        ran_key, npts, z_min, z_max, DEFAULT_COSMOLOGY
    )
    assert redshift.shape == (npts,)
    assert np.all(np.isfinite(redshift))
    assert np.all(redshift > z_min)
    assert np.all(redshift < z_max)


def test_mc_lightcone_random_ra_dec():
    ran_key = jran.key(0)
    npts = 1_000
    ra_min, ra_max = 0.0, 1.0
    dec_min, dec_max = 0.0, 0.5

    ra, dec = lcu.mc_lightcone_random_ra_dec(
        ran_key, npts, ra_min, ra_max, dec_min, dec_max
    )
    assert ra.shape == (npts,)
    assert dec.shape == (npts,)
    assert np.all(np.isfinite(ra))
    assert np.all(np.isfinite(dec))
    assert np.all(ra > ra_min)
    assert np.all(ra < ra_max)
    assert np.all(dec > dec_min)
    assert np.all(dec < dec_max)
