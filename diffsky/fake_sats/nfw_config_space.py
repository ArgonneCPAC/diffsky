"""Module generates a random 3d positions according to a triaxial NFW profile."""

import numpy as np
from jax import random as jran
from scipy import special

from .rotations3d import rotation_matrices_from_vectors
from .vector_utilities import rotate_vector_collection


def mc_ellipsoidal_positions(ran_key, rhalo, conc, major_axes, b_to_a, c_to_a):
    """Generate Monte Carlo realization of halo-centric xyz positions according to
    an ellipsoidal NFW distribution

    Parameters
    ----------
    ran_key : jax.random.key

    rhalo : ndarray, shape (n, )

    conc : ndarray, shape (n, )

    major_axes : ndarray, shape (n, 3)

    b_to_a : ndarray, shape (n, )

    c_to_a : ndarray, shape (n, )

    Returns
    -------
    pos : ndarray, shape (n, 3)

    """
    n = conc.shape[0]
    x_axes = np.tile([1, 0, 0], n).reshape((n, 3))
    rotations = rotation_matrices_from_vectors(x_axes, major_axes)

    a = rhalo / ((b_to_a * c_to_a) ** (1 / 3))
    b = a * b_to_a
    c = a * c_to_a
    pos_xyz = np.vstack(random_nfw_ellipsoid(ran_key, conc, a, b, c)).T
    pos = rotate_vector_collection(rotations, pos_xyz)
    return pos


def random_nfw_ellipsoid(ran_key, conc, a=1, b=1, c=1):
    """Generate random points within an NFW ellipsoid with unit radius.

    Parameters
    ----------
    ran_key : jax.random.key

    conc : ndarray
        Array of concentrations of shape (n, )

    a : float or ndarray of shape (n, ), optional
        Length of the x-axis. Default is 1 for a unit sphere.

    b : float or ndarray of shape (n, ), optional
        Length of the y-axis. Default is 1 for a unit sphere.

    c : float or ndarray of shape (n, ), optional
        Length of the z-axis. Default is 1 for a unit sphere.

    Returns
    -------
    x, y, z : ndarrays of shape (n, )

    """
    x, y, z = random_nfw_spherical_coords(ran_key, conc)
    return a * x, b * y, c * z


def random_nfw_spherical_coords(ran_key, conc):
    """Generate random points within an NFW sphere with unit radius.

    Parameters
    ----------
    ran_key : jax.random.key

    conc : ndarray
        Array of concentrations of shape (n, )

    Returns
    -------
    x, y, z : ndarrays of shape (n, )

    """
    conc = np.atleast_1d(conc)
    npts = conc.size

    ukey, rkey = jran.split(ran_key, 2)
    randoms = np.array(jran.uniform(ukey, shape=(3 * npts,), minval=0, maxval=1))
    r_randoms = randoms[:npts]
    xyz_randoms = randoms[npts:]
    r = random_nfw_radial_position(rkey, conc, randoms=r_randoms)
    x, y, z = _random_spherical_position(xyz_randoms)
    return r * x, r * y, r * z


def _random_spherical_position(u):
    """Generate random points on the surface of 1d sphere.

    Parameters
    ----------
    u : ndarray of shape (2*n, )
        Uniform random points in the range (0, 1)

    Returns
    -------
    x, y, z : ndarrays of shape (n, )

    """
    n = u.size
    nhalf = n // 2
    cos_t = 2 * u[:nhalf] - 1
    phi = 2 * np.pi * u[nhalf:]

    sin_t = np.sqrt((1.0 - cos_t * cos_t))

    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = cos_t

    return x, y, z


def random_nfw_radial_position(ran_key, conc, randoms=None):
    """Generate random radial positions according to an NFW profile.

    The returned positions are dimensionless, and so should be multiplied by the
    halo radius to generate halo-centric distances.

    Parameters
    ----------
    conc : ndarray
        Array of concentrations of shape (n, )

    ran_key : jax.random.key

    randoms : ndarray, optional
        Array of shape (n, ) of the CDF to use for each value of concentration

    Returns
    -------
    r : ndarray
        Array of shape (n, ) storing r/Rhalo, so that 0 < x < 1

    """
    conc = np.atleast_1d(conc)
    n = conc.size
    if randoms is None:
        u = np.array(jran.uniform(ran_key, shape=(n,), minval=0, maxval=1))
    else:
        u = np.atleast_1d(randoms)
        assert u.size == n, "randoms must have the same size as conc"
        assert np.all(randoms >= 0), "randoms must be non-negative"
        assert np.all(randoms <= 1), "randoms cannot exceed unity"

    return _qnfw(u, conc)


def _pnfwunorm(q, conc):
    """ """
    y = q * conc
    return np.log(1.0 + y) - y / (1.0 + y)


def _qnfw(p, conc):
    """ """
    assert np.all(p >= 0), "randoms must be non-negative"
    assert np.all(p <= 1), "randoms cannot exceed unity"
    p *= _pnfwunorm(1, conc)
    return (-(1.0 / np.real(special.lambertw(-np.exp(-p - 1)))) - 1) / conc
