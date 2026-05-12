""" """

import numpy as np
from jax import random as jran

from .. import ellipsoidal_nfw_phase_space as enfwps


def test_mc_ellipsoidal_nfw():
    ran_key = jran.key(0)
    n_halos = 25

    r_key, conc_key, axes_key, b_key, c_key, sigma_key, nfw_key = jran.split(ran_key, 7)

    rhalo = jran.uniform(r_key, minval=0.5, maxval=2.0, shape=(n_halos,))
    conc = jran.uniform(conc_key, minval=2.0, maxval=20.0, shape=(n_halos,))
    sigma = jran.uniform(sigma_key, minval=10.0, maxval=200.0, shape=(n_halos,))
    major_axes = jran.uniform(axes_key, minval=0.0, maxval=1.0, shape=(n_halos, 3))
    b_to_a = jran.uniform(b_key, minval=0.5, maxval=1.0, shape=(n_halos,))
    c_to_a = jran.uniform(c_key, minval=0.5, maxval=1.0, shape=(n_halos,)) * b_to_a

    pos, vel = enfwps.mc_ellipsoidal_nfw(
        nfw_key, rhalo, conc, sigma, major_axes, b_to_a, c_to_a
    )
    assert pos.shape == (n_halos, 3)
    assert vel.shape == (n_halos, 3)

    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(vel))
