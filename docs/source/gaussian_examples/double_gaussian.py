"""Demo code to fit a 2d Gaussian model with soft histograms and jax.grad"""

from jax import numpy as jnp
from collections import namedtuple
from jax import random as jran
from jax import jit as jjit
from diffsky.signdhist_lomem import nnsig_ndhist
from jax import value_and_grad

DGParams = namedtuple("DGParams", ("mu0", "sig0", "mu1", "sig1", "frac0"))

DEFAULT_PARAMS = DGParams(mu0=-1.0, sig0=0.5, mu1=1.0, sig1=1.0, frac0=0.75)
NPTS = 20_000


@jjit
def mc_double_gaussian(params, ran_key):
    u_key, n0_key, n1_key = jran.split(ran_key, 3)
    uran = jran.uniform(u_key, minval=0, maxval=1, shape=(NPTS,))
    n0 = jran.normal(n0_key, shape=(NPTS,)) * params.sig0 + params.mu0
    n1 = jran.normal(n1_key, shape=(NPTS,)) * params.sig1 + params.mu1
    mc_0 = uran < params.frac0
    xdata = jnp.where(mc_0, n0, n1)
    return xdata


@jjit
def predict_soft_xhist_mc(params, xbins, ran_key):
    xdata = mc_double_gaussian(params, ran_key)
    xhist = soft_xhist(xdata, xbins)
    return xhist


@jjit
def predict_soft_xhist_weighted(params, xbins, ran_key):
    n0_key, n1_key = jran.split(ran_key, 2)
    n0 = jran.normal(n0_key, shape=(NPTS,)) * params.sig0 + params.mu0
    n1 = jran.normal(n1_key, shape=(NPTS,)) * params.sig1 + params.mu1
    xhist0 = soft_xhist(n0, xbins)
    xhist1 = soft_xhist(n1, xbins)
    xhist = params.frac0 * xhist0 + (1.0 - params.frac0) * xhist1
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
def weighted_mae_loss(params, loss_data):
    xhist_target, xbins, ran_key = loss_data
    xhist_pred = predict_soft_xhist_weighted(params, xbins, ran_key)
    loss = _mae_kern(xhist_pred, xhist_target)
    return loss


@jjit
def mc_mae_loss(params, loss_data):
    xhist_target, xbins, ran_key = loss_data
    xhist_pred = predict_soft_xhist_mc(params, xbins, ran_key)
    loss = _mae_kern(xhist_pred, xhist_target)
    return loss


@jjit
def param_update(params, grads, learning_rate):
    """Update namedtuple params by taking a small step down the gradient"""
    new_params = params._make(jnp.array(params) - jnp.array(grads) * learning_rate)
    return new_params


weighted_mae_loss_and_grad = jjit(value_and_grad(weighted_mae_loss, argnums=0))
mc_mae_loss_and_grad = jjit(value_and_grad(mc_mae_loss, argnums=0))
