"""Generate an ellipsoidal velocity distribution"""

import numpy as np
from jax import random as jran

from .rotations3d import rotation_matrices_from_vectors
from .vector_utilities import rotate_vector_collection

NEWTON_G = 4.3e-09  # (Mpc/Msun)*(km/s)^2


def calculate_virial_velocity(halo_mass, halo_radius):
    """Calculate halo virial velocity to set normalization
    of NFW satellite velocity dispersion

    Parameters
    ----------
    halo_mass : ndarray, shape (n, )
        Units of Msun

    halo_radius : ndarray, shape (n, )
        Units of Mpc

    Returns
    -------
    halo_vvir : ndarray, shape (n, )
        Units of km/s

    """
    halo_vvir = np.sqrt(NEWTON_G * halo_mass / halo_radius)
    return halo_vvir


def mc_ellipsoidal_velocities(ran_key, sigma, major_axes, b_to_a, c_to_a):
    """Generate a population with ellipsoidal velocities aligned with the major axes

    Parameters
    ----------
    ran_key : jax.random.PRNGKey object

    sigma : ndarray of shape (n, )

    major_axes : ndarray of shape (n, 3)
        xyz coordinates of the major axis of each ellipse
        Note that the normalization of the input major_axes will be ignored

    b_to_a : ndarray of shape (n, )

    c_to_a : ndarray of shape (n, )

    Returns
    -------
    vel : ndarray of shape (n, 3)

    """
    n = sigma.shape[0]
    x_axes = np.tile([1, 0, 0], n).reshape((n, 3))
    rotations = rotation_matrices_from_vectors(x_axes, major_axes)
    vel_xyz = mc_cartesian_ellipsoidal_velocities(ran_key, sigma, b_to_a, c_to_a)
    vel = rotate_vector_collection(rotations, vel_xyz)
    return vel


def mc_cartesian_ellipsoidal_velocities(ran_key, sigma, b_to_a, c_to_a):
    """Generate a population with ellipsoidal velocities aligned with the Cartesian axes

    Parameters
    ----------
    ran_key : jax.random.PRNGKey object

    sigma : ndarray of shape (n, )

    b_to_a : ndarray of shape (n, )

    c_to_a : ndarray of shape (n, )

    Returns
    -------
    w : ndarray of shape (n, 3)

    """
    xkey, ykey, zkey, ran_key = jran.split(ran_key, 4)

    n = sigma.shape[0]
    vx_u = jran.normal(xkey, (n,))
    vy_u = jran.normal(ykey, (n,))
    vz_u = jran.normal(zkey, (n,))

    vx = vx_u * sigma / np.sqrt(3)
    vy = vy_u * sigma * b_to_a / np.sqrt(3)
    vz = vz_u * sigma * c_to_a / np.sqrt(3)
    v = np.array((vx, vy, vz)).T

    sigma2_x = sigma / np.sqrt(3)
    sigma2_y = b_to_a * sigma / np.sqrt(3)
    sigma2_z = c_to_a * sigma / np.sqrt(3)
    sigma_ellipse = np.sqrt(sigma2_x**2 + sigma2_y**2 + sigma2_z**2)
    volume_ratio = sigma / sigma_ellipse
    return v * volume_ratio.reshape((n, 1))
