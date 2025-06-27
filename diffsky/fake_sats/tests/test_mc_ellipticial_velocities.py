"""
"""
import numpy as np
from jax import random as jran
from ..ellipsoidal_nfw_phase_space import mc_ellipsoidal_nfw


def test_mc_ellipsoidal_nfw():
    ran_key = jran.PRNGKey(0)
    n = 500
    rvir = np.zeros(n) + 0.5
    conc = np.random.uniform(2, 10, n)
    sigma = np.random.normal(loc=0, scale=200, size=n)
    major_axes = np.random.uniform(-1, 1, 3 * n).reshape((n, 3))
    b_to_a = np.random.uniform(0, 1, n)
    c_to_a = np.random.uniform(0, 1, n) * b_to_a
    pos, vel = mc_ellipsoidal_nfw(
        ran_key, rvir, conc, sigma, major_axes, b_to_a, c_to_a
    )
