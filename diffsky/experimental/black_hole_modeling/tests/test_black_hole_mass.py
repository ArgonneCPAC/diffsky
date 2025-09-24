import numpy as np
from jax import random as jran

from .. import black_hole_mass as bhm


def test_bh_mass_from_bulge_mass():
    n_halos = 2_000
    bulge_mass = np.logspace(7, 11, n_halos)
    bh_mass = bhm.bh_mass_from_bulge_mass(bulge_mass)
    assert np.all(np.isfinite(bh_mass))
    assert bh_mass.shape == (n_halos,)
    assert np.all(bh_mass < bulge_mass)
    assert np.all(bh_mass > 1e-4 * bulge_mass)


def test_monte_carlo_black_hole_mass():
    ran_key = jran.key(0)
    n_halos = 2_000
    bulge_mass = np.logspace(7, 11, n_halos)
    bh_mass = bhm.monte_carlo_black_hole_mass(bulge_mass, ran_key)
    assert np.all(np.isfinite(bh_mass))
    assert bh_mass.shape == (n_halos,)
    assert np.all(bh_mass < bulge_mass)
    assert np.all(bh_mass > 1e-4 * bulge_mass)
