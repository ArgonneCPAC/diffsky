""" """

import os

import numpy as np
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY

from .. import halo_boundary_functions as hbf

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DRN_TESTING_DATA = os.path.join(_THIS_DRNAME, "testing_data")


def test_density_threshold():
    cosmo_params = DEFAULT_COSMOLOGY._make((0.3075, -1.0, 0.0, 0.6774))
    n_halos = 500
    redshift = np.linspace(0, 5, n_halos)
    rho_thresh_200c = hbf.density_threshold(cosmo_params, redshift, "200c")
    assert np.all(np.isfinite(rho_thresh_200c))

    rho_thresh_200m = hbf.density_threshold(cosmo_params, redshift, "200m")
    assert np.all(np.isfinite(rho_thresh_200m))


def test_halo_mass_to_halo_radius():

    fn_halo_mass = os.path.join(DRN_TESTING_DATA, "halo_mass_ht_test.txt")
    halo_mass_ht = np.loadtxt(fn_halo_mass)

    fn_redshift = os.path.join(DRN_TESTING_DATA, "redshift_ht_test.txt")
    redshift_ht = np.loadtxt(fn_redshift)

    fn_halo_radius = os.path.join(DRN_TESTING_DATA, "halo_radius_200c_ht_test.txt")
    halo_radius_h1p0_ht = np.loadtxt(fn_halo_radius)

    mdef = "200c"

    cosmo_params = DEFAULT_COSMOLOGY._make((0.3075, -1.0, 0.0, 0.6774))

    halo_radius = hbf.halo_mass_to_halo_radius(
        halo_mass_ht, cosmo_params, redshift_ht, mdef
    )

    h_rescaling_factor = cosmo_params.h ** (-2 / 3)
    halo_radius_h1p0 = halo_radius / h_rescaling_factor

    fracerr = (halo_radius_h1p0 - halo_radius_h1p0_ht) / halo_radius_h1p0_ht

    assert np.allclose(fracerr, 0.0, atol=0.01)
