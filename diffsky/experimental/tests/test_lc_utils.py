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


def test_get_z_obs_from_z_true():
    ran_key = jran.key(0)
    n = 5_000
    ran_key, z_key, v_key = jran.split(ran_key, 3)

    z_true = jran.uniform(z_key, minval=0, maxval=5, shape=(n,))
    v_pec_kms = jran.uniform(v_key, minval=-500, maxval=500, shape=(n,))

    z_obs = lcu.get_z_obs_from_z_true(z_true, v_pec_kms)

    dz = z_obs - z_true
    assert np.allclose(np.mean(dz), 0.0, atol=1e-3)
    assert np.std(z_obs - z_true) < 0.01


def test_consistent_ra_dec_theta_phi_minmax():
    ran_key = jran.key(0)

    n_tests = 1_000
    for __ in range(n_tests):
        ran_key, theta_key, phi_key = jran.split(ran_key, 3)
        theta_min, theta_max = np.sort(
            jran.uniform(theta_key, minval=0, maxval=np.pi, shape=(2,))
        )
        phi_min, phi_max = np.sort(
            jran.uniform(phi_key, minval=0, maxval=2 * np.pi, shape=(2,))
        )
        ra_min, ra_max, dec_min, dec_max = lcu._get_ra_dec_minmax_from_theta_phi_minmax(
            theta_min, theta_max, phi_min, phi_max
        )

        assert np.all(ra_min < ra_max)
        assert np.all(ra_min >= 0.0)
        assert np.all(ra_min <= 360.0)
        assert np.all(ra_max >= 0.0)
        assert np.all(ra_max <= 360.0)

        assert np.all(dec_min < dec_max)
        assert np.all(dec_min >= -90.0)
        assert np.all(dec_min <= 90.0)
        assert np.all(dec_max >= -90.0)
        assert np.all(dec_max <= 90.0)

        theta_min2, theta_max2, phi_min2, phi_max2 = (
            lcu._get_theta_phi_minmax_from_ra_dec_minmax(
                ra_min, ra_max, dec_min, dec_max
            )
        )
        assert np.allclose(theta_min, theta_min2, rtol=1e-4)
        assert np.allclose(theta_max, theta_max2, rtol=1e-4)
        assert np.allclose(phi_min, phi_min2, rtol=1e-4)
        assert np.allclose(phi_max, phi_max2, rtol=1e-4)


def test_mc_lightcone_random_theta_phi():
    n_tests = 100
    npts = 2_000
    ran_key = jran.key(0)
    for __ in range(n_tests):
        ran_key, theta_key, phi_key = jran.split(ran_key, 3)
        theta_min, theta_max = np.sort(
            jran.uniform(theta_key, minval=0, maxval=np.pi, shape=(2,))
        )
        phi_min, phi_max = np.sort(
            jran.uniform(phi_key, minval=0, maxval=2 * np.pi, shape=(2,))
        )

        mc_theta, mc_phi = lcu.mc_lightcone_random_theta_phi(
            ran_key, npts, theta_min, theta_max, phi_min, phi_max
        )

        assert np.all(mc_theta > theta_min)
        assert np.all(mc_theta < theta_max)

        assert np.all(mc_phi > phi_min)
        assert np.all(mc_phi < phi_max)
