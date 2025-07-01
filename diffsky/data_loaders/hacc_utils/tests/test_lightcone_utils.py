""""""

import os

import numpy as np
import pytest
from dsps.cosmology import flat_wcdm
from jax import random as jran

from .. import lightcone_utils as hlu

try:
    from haccytrees import Simulation as HACCSim

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False
NO_HACC_MSG = "Must have haccytrees installed to run this test"


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DRN_TESTING_DATA = os.path.join(_THIS_DRNAME, "testing_data")


def test_jnp_take_matrix():
    matrix = np.tile((0, 1), 5).reshape((5, 2))
    indxarr = np.array((0, 1, 0, 1, 1))
    arr = hlu.jnp_take_matrix(matrix, indxarr)
    assert np.allclose(arr, indxarr)


def test_read_lc_ra_dec_patch_decomposition():
    fn = os.path.join(DRN_TESTING_DATA, "lc_cores-decomposition.txt")
    _res = hlu.read_lc_ra_dec_patch_decomposition(fn)
    patch_decomposition, sky_frac, solid_angles = _res
    assert np.all(sky_frac > 0)
    assert np.all(sky_frac < 1)
    assert np.all(solid_angles > 0)
    assert np.all(solid_angles < hlu.SQDEG_OF_SPHERE)

    assert np.allclose(sky_frac, sky_frac[0], rtol=0.01)
    assert np.allclose(solid_angles, solid_angles[0], rtol=0.01)


def test_calculate_solid_angle():
    ra_min, ra_max = 0, np.pi
    dec_min, dec_max = 0, np.pi / 2
    solid_angle, fsky = hlu.calculate_solid_angle(ra_min, ra_max, dec_min, dec_max)
    assert np.allclose(fsky, 0.25)
    assert np.allclose(fsky * hlu.SQDEG_OF_SPHERE, solid_angle, rtol=1e-3)

    ra_min, ra_max = np.pi, 2 * np.pi
    dec_min, dec_max = np.pi / 2, np.pi
    solid_angle, fsky = hlu.calculate_solid_angle(ra_min, ra_max, dec_min, dec_max)
    assert np.allclose(fsky, 0.25)
    assert np.allclose(fsky * hlu.SQDEG_OF_SPHERE, solid_angle, rtol=1e-3)

    ra_min, ra_max = 0, 2 * np.pi
    dec_min, dec_max = np.pi / 2, np.pi
    solid_angle, fsky = hlu.calculate_solid_angle(ra_min, ra_max, dec_min, dec_max)
    assert np.allclose(fsky, 0.5)
    assert np.allclose(fsky * hlu.SQDEG_OF_SPHERE, solid_angle, rtol=1e-3)


def test_compute_theta_phi_agrees_with_haccytrees():
    """Small dataset taken from LastJourney/lc_cores-266.0.hdf5"""
    x_haccytrees_tdata = np.loadtxt(
        os.path.join(DRN_TESTING_DATA, "x_haccytrees_tdata.txt")
    )
    y_haccytrees_tdata = np.loadtxt(
        os.path.join(DRN_TESTING_DATA, "y_haccytrees_tdata.txt")
    )
    z_haccytrees_tdata = np.loadtxt(
        os.path.join(DRN_TESTING_DATA, "z_haccytrees_tdata.txt")
    )

    theta_haccytrees_tdata = np.loadtxt(
        os.path.join(DRN_TESTING_DATA, "theta_haccytrees_tdata.txt")
    )
    phi_haccytrees_tdata = np.loadtxt(
        os.path.join(DRN_TESTING_DATA, "phi_haccytrees_tdata.txt")
    )
    # Sanity check range on tdata
    assert np.all(theta_haccytrees_tdata > 0)
    assert np.all(theta_haccytrees_tdata < np.pi)
    assert np.all(phi_haccytrees_tdata > 0)
    assert np.all(phi_haccytrees_tdata < 2 * np.pi)

    theta_recomputed, phi_recomputed = hlu.get_theta_phi(
        x_haccytrees_tdata, y_haccytrees_tdata, z_haccytrees_tdata
    )

    assert np.allclose(theta_recomputed, theta_haccytrees_tdata, rtol=1e-3)
    assert np.allclose(phi_recomputed, phi_haccytrees_tdata, rtol=1e-3)


def test_ra_dec_range():
    ran_key = jran.key(0)
    n_tests = 10
    n = 5_000
    for __ in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        pos = jran.uniform(test_key, minval=-100, maxval=100, shape=(n, 3))
        ra, dec = hlu.get_ra_dec(pos[:, 0], pos[:, 1], pos[:, 2])
        assert np.all(ra > 0)
        assert np.all(ra < 360.0)

        assert np.all(dec > -90.0)
        assert np.all(dec < 90.0)


def test_theta_phi_range():
    ran_key = jran.key(0)
    n_tests = 10
    n = 5_000
    for __ in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        pos = jran.uniform(test_key, minval=-100, maxval=100, shape=(n, 3))
        theta, phi = hlu.get_theta_phi(pos[:, 0], pos[:, 1], pos[:, 2])
        assert np.all(theta > 0)
        assert np.all(theta < np.pi)

        assert np.all(phi > 0)
        assert np.all(phi < 2 * np.pi)


@pytest.mark.skipif(not HAS_HACCYTREES, reason=NO_HACC_MSG)
def test_compute_redshift_agrees_with_haccytrees():
    """Small dataset taken from LastJourney/lc_cores-266.0.hdf5"""
    # haccytrees coords are in mpc/h
    x = np.loadtxt(os.path.join(DRN_TESTING_DATA, "x_haccytrees_tdata.txt"))
    y = np.loadtxt(os.path.join(DRN_TESTING_DATA, "y_haccytrees_tdata.txt"))
    z = np.loadtxt(os.path.join(DRN_TESTING_DATA, "z_haccytrees_tdata.txt"))

    redshift = np.loadtxt(
        os.path.join(DRN_TESTING_DATA, "redshift_haccytrees_tdata.txt")
    )

    sim = HACCSim.simulations["LastJourney"]
    cosmo_params = flat_wcdm.CosmoParams(
        *(sim.cosmo.Omega_m, sim.cosmo.w0, sim.cosmo.wa, sim.cosmo.h)
    )
    redshift_recomputed = hlu.get_redshift_from_xyz(x, y, z, cosmo_params)

    assert np.allclose(redshift, redshift_recomputed, atol=0.001)
    assert np.allclose(redshift, redshift_recomputed, atol=0.001)
