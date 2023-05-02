"""
"""
from jax import numpy as jnp
from jax import vmap
from jax import jit as jjit
from jax import lax


@jjit
def _tw_cuml_lax_kern(x, m, h):
    """Triweight kernel version of an err function."""
    z = (x - m) / h
    val = (
        -5 * z**7 / 69984
        + 7 * z**5 / 2592
        - 35 * z**3 / 864
        + 35 * z / 96
        + 1 / 2
    )
    val = lax.cond(z < -3, lambda s: 0.0, lambda s: val, z)
    val = lax.cond(z > 3, lambda s: 1.0, lambda s: val, z)
    return val


@jjit
def _tw_cuml_kern(x, m, h):
    """Triweight kernel version of an err function."""
    z = (x - m) / h
    val = (
        -5 * z**7 / 69984
        + 7 * z**5 / 2592
        - 35 * z**3 / 864
        + 35 * z / 96
        + 1 / 2
    )
    val = jnp.where(z < -3, 0, val)
    val = jnp.where(z > 3, 1, val)
    return val


@jjit
def _tw_bin_weight_kern(x, sig, lo, hi):
    """Triweight kernel integrated across the boundaries of a single bin."""
    a = _tw_cuml_kern(x, lo, sig)
    b = _tw_cuml_kern(x, hi, sig)
    return a - b


@jjit
def _tw_bin_weight_lax_kern(x, sig, lo, hi):
    """Triweight kernel integrated across the boundaries of a single bin."""
    a = _tw_cuml_lax_kern(x, lo, sig)
    b = _tw_cuml_lax_kern(x, hi, sig)
    return a - b


_tw_hist1d_vmap = jjit(vmap(_tw_bin_weight_kern, in_axes=(None, None, 0, 0)))


@jjit
def tw_hist1d(x, sig, xbins):
    return jnp.sum(_tw_hist1d_vmap(x, sig, xbins[:-1], xbins[1:]), axis=1)


_A = (0, 0, 0, 0)
_tw_bin_weight_lax_kern_vmap = jjit(vmap(_tw_bin_weight_lax_kern, in_axes=_A))


@jjit
def _tw_ndhist_kern(nddata, ndsig, ndlo, ndhi):
    return jnp.prod(_tw_bin_weight_lax_kern_vmap(nddata, ndsig, ndlo, ndhi))


_tw_ndhist_kern_vmap = jjit(vmap(_tw_ndhist_kern, in_axes=(0, 0, None, None)))


@jjit
def _tw_ndhist_sumkern(nddata, ndsig, ndlo, ndhi):
    return jnp.sum(_tw_ndhist_kern_vmap(nddata, ndsig, ndlo, ndhi))


_tw_ndhist_vmap = jjit(vmap(_tw_ndhist_sumkern, in_axes=(None, None, 0, 0)))


@jjit
def tw_ndhist(nddata, ndsig, ndbins_lo, ndbins_hi):
    """N-dimensional weighted histogram with arbitrary bins

    Parameters
    ----------
    nddata : ndarray of shape (npts, ndim)
        Collection of npts data points residing in an ndim-dimensional space

    ndsig : ndarray of shape (npts, ndim)
        Triweight scatter for each point in each dimension

    ndbins_lo : ndarray of shape (nbins, ndim)
        Lower bound in each dimension for each bin

    ndbins_hi : ndarray of shape (nbins, ndim)
        Upper bound in each dimension for each bin

    Returns
    -------
    ndhist : ndarray of shape (nbins, )
        Weighted histogram of nddata

    """
    return _tw_ndhist_vmap(nddata, ndsig, ndbins_lo, ndbins_hi)
