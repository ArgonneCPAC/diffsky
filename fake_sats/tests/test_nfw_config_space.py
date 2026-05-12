""""""

import numpy as np
from jax import random as jran

from .. import nfw_config_space as nfwcs


def test_mc_ellipsoidal_positions_are_inside_the_input_halos():
    """Enforce that the mc_ellipsoidal_positions function returns a collection of points
    that are located within the halo"""
    ran_key = jran.key(0)
    n_halos = 25

    r_key, conc_key, axes_key, b_key, c_key, pos_key = jran.split(ran_key, 6)

    rhalo = jran.uniform(r_key, minval=0.5, maxval=2.0, shape=(n_halos,))
    conc = jran.uniform(conc_key, minval=2.0, maxval=20.0, shape=(n_halos,))
    major_axes = jran.uniform(axes_key, minval=0.0, maxval=1.0, shape=(n_halos, 3))
    b_to_a = jran.uniform(b_key, minval=0.5, maxval=1.0, shape=(n_halos,))
    c_to_a = jran.uniform(c_key, minval=0.5, maxval=1.0, shape=(n_halos,)) * b_to_a

    pos = nfwcs.mc_ellipsoidal_positions(
        pos_key, rhalo, conc, major_axes, b_to_a, c_to_a
    )
    assert np.all(np.isfinite(pos))
    assert pos.shape == (n_halos, 3)
    assert np.all(pos >= -rhalo.reshape((-1, 1)))
    assert np.all(pos <= rhalo.reshape((-1, 1)))
    assert ~np.any(pos == 0.0)
