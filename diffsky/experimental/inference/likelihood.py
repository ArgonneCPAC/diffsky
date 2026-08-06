"""
These are auxiliary functions just to demonstrate how to use the code.
They were taken from diffsky's docs tutorial:
"supplementary source code for doft histograms."
"""

import jax
import jax.numpy as jnp

from diffsky.soft_histograms.signdhist_lomem import nnsig_ndhist, nnsig_ndhist_weighted


@jax.jit
def _mae_kern(x, y):
    """Mean absolute error"""
    abs_diff = jnp.abs(y - x)
    return jnp.mean(abs_diff)


@jax.jit
def _mse_kern(x, y):
    """Mean squared error"""
    sq_diff = (y - x) ** 2
    return jnp.mean(sq_diff)


@jax.jit
def _poisson_kern(x, y):
    """log of poisson distribution
    y: target
    x: pred.
    """
    return jnp.sum(y * jnp.log(x) - x)


@jax.jit
def soft_xhist(xdata, xbins):
    """Soft histogram function
    This is a wrapper around diffsky.nnsig_ndhist for 1d data"""
    nbins = xbins.shape[0]
    xbins_lo = xbins[:-1].reshape((nbins - 1, 1))
    xbins_hi = xbins[1:].reshape((nbins - 1, 1))
    dx = jnp.diff(xbins).mean()
    ndsig = jnp.zeros_like(xbins_lo) + dx / 2
    xdata = xdata.reshape((-1, 1))
    xhist = nnsig_ndhist(xdata, ndsig, xbins_lo, xbins_hi)
    return xhist
