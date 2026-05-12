""" """

from jax import random as jran

from .ellipsoidal_velocities import mc_ellipsoidal_velocities
from .nfw_config_space import mc_ellipsoidal_positions


def mc_ellipsoidal_nfw(ran_key, rhalo, conc, sigma, major_axes, b_to_a, c_to_a):
    """Generate points in the phase space of an ellipsoidal NFW halo

    Parameters
    ----------
    ran_key : jax.random.key

    rhalo : ndarray of shape (npts, )

    conc : ndarray of shape (npts, )

    sigma : ndarray of shape (npts, )

    major_axes : ndarray of shape (n, 3)
        xyz coordinates of the major axis of each ellipse

    b_to_a : ndarray of shape (n, )

    c_to_a : ndarray of shape (n, )

    Returns
    -------
    pos : ndarray of shape (npts, 3)

    vel : ndarray of shape (npts, 3)

    """
    pos_key, vel_key = jran.split(ran_key, 2)
    pos = mc_ellipsoidal_positions(pos_key, rhalo, conc, major_axes, b_to_a, c_to_a)
    vel = mc_ellipsoidal_velocities(vel_key, sigma, major_axes, b_to_a, c_to_a)
    return pos, vel
