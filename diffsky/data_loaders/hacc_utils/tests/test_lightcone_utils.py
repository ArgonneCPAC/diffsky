""""""

import os

import numpy as np

from .. import lightcone_utils as hlu

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
