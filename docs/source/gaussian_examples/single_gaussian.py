"""Demo code to fit a 1d Gaussian model with soft histograms and jax.grad"""

from jax import numpy as jnp
from jax import jit as jjit
from jax import random as jran
from jax import value_and_grad
from collections import namedtuple
from diffsky.signdhist_lomem import nnsig_ndhist

GParams = namedtuple("GParams", ("mu", "sig"))
DEFAULT_PARAMS = GParams(mu=-1.0, sig=1.0)

NPTS = 20_000


@jjit
def mc_single_gaussian(params, ran_key):
    """Draw a Monte Carlo realization of a Gaussian"""
    xdata = jran.normal(ran_key, shape=(NPTS,)) * params.sig + params.mu
    return xdata


@jjit
def mc_predict_hard_edged_xhist(params, xbins, ran_key):
    """Predict histogram counts by applying jnp.histogram to
    a Monte Carlo realization of a Gaussian"""
    xdata = mc_single_gaussian(params, ran_key)
    xhist, __ = jnp.histogram(xdata, bins=xbins)
    return xhist


@jjit
def mc_predict_soft_xhist(params, xbins, ran_key):
    """Predict histogram counts by applying a soft histogram to
    a Monte Carlo realization of a Gaussian"""
    xdata = mc_single_gaussian(params, ran_key)
    n = xdata.shape[0]
    xdata = xdata.reshape((n, 1))
    xhist = soft_xhist(xdata, xbins)
    return xhist


@jjit
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


@jjit
def _mae_kern(x, y):
    """Mean absolute error"""
    abs_diff = jnp.abs(y - x)
    return jnp.mean(abs_diff)


@jjit
def hard_edged_xhist_loss(params, loss_data):
    """Loss function based on a histogram with hard-edged bins"""
    xhist_target, xbins, ran_key = loss_data
    xhist_pred = mc_predict_hard_edged_xhist(params, xbins, ran_key)
    loss = _mae_kern(xhist_pred, xhist_target)
    return loss


@jjit
def soft_xhist_loss(params, loss_data):
    """Loss function based on a soft histogram"""
    xhist_target, xbins, ran_key = loss_data
    xhist_pred = mc_predict_soft_xhist(params, xbins, ran_key)
    loss = _mae_kern(xhist_pred, xhist_target)
    return loss


@jjit
def param_update(params, grads, learning_rate):
    """Update namedtuple params by taking a small step down the gradient"""
    new_params = params._make(jnp.array(params) - jnp.array(grads) * learning_rate)
    return new_params


hard_edged_xhist_loss_and_grad = jjit(value_and_grad(hard_edged_xhist_loss, argnums=0))
soft_xhist_loss_and_grad = jjit(value_and_grad(soft_xhist_loss, argnums=0))
