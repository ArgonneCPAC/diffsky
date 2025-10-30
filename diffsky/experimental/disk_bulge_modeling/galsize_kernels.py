"""Module implementing fitting functions for the size-luminosity relation
taken from Zhang & Yang (2017), arXiv:1707.04979.
"""

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

LITTLEH = 0.7
HSQ = LITTLEH * LITTLEH

DiskSizeParams = namedtuple(
    "DiskSizeParams", ("disk_alpha", "disk_beta", "disk_gamma", "disk_mzero")
)
BulgeSizeParams = namedtuple(
    "BulgeSizeParams", ("bulge_alpha", "bulge_beta", "bulge_gamma", "bulge_mzero")
)

DISK_SIZE_PARAMS = DiskSizeParams(
    disk_alpha=0.22, disk_beta=1.24, disk_gamma=8.83, disk_mzero=4.49e11 / HSQ
)
BULGE_SIZE_PARAMS = BulgeSizeParams(
    bulge_alpha=0.11, bulge_beta=0.60, bulge_gamma=1.75, bulge_mzero=1.35e10 / HSQ
)
R50_MIN, R50_MAX = 0.1, 40.0

R50_SCATTER = 0.2


@jjit
def mc_r50_disk(mstar, redshift, ran_key):
    logr50_med = jnp.log10(_r50_disk_kern(mstar, redshift))
    logr50 = jran.normal(ran_key, shape=logr50_med.shape) * R50_SCATTER + logr50_med
    r50 = 10**logr50
    r50 = jnp.clip(r50, R50_MIN, R50_MAX)
    return r50


@jjit
def mc_r50_bulge(mstar, redshift, ran_key):
    logr50_med = jnp.log10(_r50_bulge_kern(mstar, redshift))
    logr50 = jran.normal(ran_key, shape=logr50_med.shape) * R50_SCATTER + logr50_med
    r50 = 10**logr50
    r50 = jnp.clip(r50, R50_MIN, R50_MAX)
    return r50


@jjit
def _r50_disk_kern(mstar, redshift):
    r50_z0 = _r50_vs_mstar_disk(mstar)
    shrinking_factor = _redshift_shrinking_factor(redshift)
    r50 = r50_z0 * shrinking_factor
    r50 = jnp.clip(r50, R50_MIN, R50_MAX)
    return r50


@jjit
def _r50_bulge_kern(mstar, redshift):
    r50_z0 = _r50_vs_mstar_bulge(mstar)
    shrinking_factor = _redshift_shrinking_factor(redshift)
    r50 = r50_z0 * shrinking_factor
    r50 = jnp.clip(r50, R50_MIN, R50_MAX)
    return r50


@jjit
def _redshift_shrinking_factor(redshift, z0=1.0, ymin=1.0, ymax=2.0, k=4.0):
    """Sigmoid function calibrated against van der Wel 2014."""
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (redshift - z0)))


@jjit
def _r50_vs_mstar_singlez_kern(mstar, alpha, beta, gamma, mzero):
    x = mstar / mzero
    r50 = gamma * (x**alpha) * (1 + x) ** (beta - alpha)
    return r50


@jjit
def _r50_vs_mstar_disk(mstar, disk_size_params=DISK_SIZE_PARAMS):
    return _r50_vs_mstar_singlez_kern(mstar, *disk_size_params)


@jjit
def _r50_vs_mstar_bulge(mstar, bulge_size_params=BULGE_SIZE_PARAMS):
    return _r50_vs_mstar_singlez_kern(mstar, *bulge_size_params)
