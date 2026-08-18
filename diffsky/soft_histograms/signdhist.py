"""JAX kernels for N-dimensional triweighted histograms"""

from jax import numpy as jnp
from jax import vmap
from jax import jit as jjit
from jax.nn import sigmoid as nn_sigmoid


@jjit
def _nn_sigmoid(x, m, h):
    """jax.nn sigmoid version of an err function.
    This kernel accepts and returns scalars for all arguments
    """
    z = (x - m) / h
    return nn_sigmoid(z)


@jjit
def _nnsig_bin_weight_kern(x, sig, lo, hi):
    """jax.nn sigmoid kernel integrated across the boundaries of a single bin.

    This kernel accepts and returns scalars for all arguments
    """
    a = _nn_sigmoid(x, lo, sig)
    b = _nn_sigmoid(x, hi, sig)
    return a - b


# vmap each individual scalar point to be an individual point in n-dimensions
_A = (0, 0, 0, 0)
_nnsig_bin_weight_kern_vmap = jjit(vmap(_nnsig_bin_weight_kern, in_axes=_A))


@jjit
def _nnsig_ndhist_kern(nddata, ndsig, ndlo, ndhi):
    """
    For an individual scalar point in n-dimensions
    we calculate w: the product of the weights across all dimensions
    for a single n-dimensional bin
    """
    w = jnp.prod(_nnsig_bin_weight_kern_vmap(nddata, ndsig, ndlo, ndhi))
    return w


# Vectorize from a single n-dimensional point to many n-dimensional points
_nnsig_ndhist_kern_vmap = jjit(vmap(_nnsig_ndhist_kern, in_axes=(0, 0, None, None)))


@jjit
def _nnsig_ndhist_sumkern(nddata, ndsig, ndlo, ndhi):
    """Sum contributions from all the n-dimensional points
    again for a single n-dimensional bin
    """
    return jnp.sum(_nnsig_ndhist_kern_vmap(nddata, ndsig, ndlo, ndhi))


# Repeat the _nnsig_ndhist_sumkern kernel for many bins at once
_nnsig_ndhist_vmap = jjit(vmap(_nnsig_ndhist_sumkern, in_axes=(None, None, 0, 0)))


@jjit
def nnsig_ndhist(nddata, ndsig, ndbins_lo, ndbins_hi):
    """N-dimensional weighted histogram with arbitrary bins

    Parameters
    ----------
    nddata : ndarray of shape (npts, ndim)
        Collection of npts data points residing in an ndim-dimensional space

    ndsig : ndarray of shape (npts, ndim)
        Sigmoid scatter for each point in each dimension

    ndbins_lo : ndarray of shape (nbins, ndim)
        Lower bound in each dimension for each bin

    ndbins_hi : ndarray of shape (nbins, ndim)
        Upper bound in each dimension for each bin

    Returns
    -------
    ndhist : ndarray of shape (nbins, )
        Weighted histogram of nddata

    Notes
    -----
    The nnsig_ndhist function can be used to calculate quantities such as
    the number density of points that fall within a cell of N-dimensional data

    """
    return _nnsig_ndhist_vmap(nddata, ndsig, ndbins_lo, ndbins_hi)


@jjit
def _nnsig_ndhist_weighted_sum_kern(nddata, ndsig, y, ndlo, ndhi):
    """
    For an individual scalar point in n-dimensions
    we calculate w * y: where w is the product of the weights across all dimensions
    for a single n-dimensional bin, and y is the quantity we are summing
    """
    w = jnp.prod(_nnsig_bin_weight_kern_vmap(nddata, ndsig, ndlo, ndhi))
    return w * y


# Vectorize from a single n-dimensional point to many n-dimensional points
_nnsig_ndhist_weighted_sum_vmap = jjit(
    vmap(_nnsig_ndhist_weighted_sum_kern, in_axes=(0, 0, 0, None, None))
)


@jjit
def _nnsig_ndhist_weighted_kern(nddata, ndsig, y, ndlo, ndhi):
    """Sum contributions from all the n-dimensional points
    again for a single n-dimensional bin
    """
    return jnp.sum(_nnsig_ndhist_weighted_sum_vmap(nddata, ndsig, y, ndlo, ndhi))


# Repeat the _nnsig_ndhist_weighted_kern kernel for many bins at once
_nnsig_ndhist_weighted_vmap = jjit(
    vmap(_nnsig_ndhist_weighted_kern, in_axes=(None, None, None, 0, 0))
)


@jjit
def nnsig_ndhist_weighted(nddata, ndsig, ydata, ndbins_lo, ndbins_hi):
    """Calculate sum of ydata for those points in nddata
    that fall within arbitrary N-dimensional bins

    Parameters
    ----------
    nddata : ndarray of shape (npts, ndim)
        Collection of npts data points residing in an ndim-dimensional space

    ndsig : ndarray of shape (npts, ndim)
        Sigmoid scatter for each point in each dimension

    ydata : ndarray of shape (npts, )
        Quantity to be summed

    ndbins_lo : ndarray of shape (nbins, ndim)
        Lower bound in each dimension for each bin

    ndbins_hi : ndarray of shape (nbins, ndim)
        Upper bound in each dimension for each bin

    Returns
    -------
    ndhist : ndarray of shape (nbins, )
        Weighted histogram of nddata

    Notes
    -----
    The nnsig_ndhist_weighted function and the nnsig_ndhist together
    can be used to calculate quantities such as < y | x0, x1, ..., x2 >,
    the average value of y for those points x that fall
    within a cell of N-dimensional data

    """
    return _nnsig_ndhist_weighted_vmap(nddata, ndsig, ydata, ndbins_lo, ndbins_hi)
