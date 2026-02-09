""""""

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ...utils import _sig_slope, _sigmoid

R50_MIN, R50_MAX = 0.1, 40.0
R50_SCATTER = 0.2

XTP = 10.0
LGM_LGR_SLOPE_K_BULGE = 1.0
YTP_Z_K_BULGE = 2.5
SLOPE_Z_K_BULGE = 5.0


DEFAULT_BULGE_SIZE_PDICT = dict(
    z_x0_x0=1.0,
    z_x0_lo=10.2,
    z_x0_hi=11.6,
    lgm_ytp=0.3,
    lgm_slope_lo=0.14,
    lgm_slope_hi=0.4,
)
BulgeSizeParams = namedtuple("BulgeSizeParams", list(DEFAULT_BULGE_SIZE_PDICT.keys()))
DEFAULT_BULGE_SIZE_PARAMS = BulgeSizeParams(**DEFAULT_BULGE_SIZE_PDICT)


def mc_r50_bulge_size(logsm, redshift, ran_key, bulge_params=DEFAULT_BULGE_SIZE_PARAMS):
    """Monte Carlo realization of size--mass relation for bulges

    Parameters
    ----------
    logsm: array, shape (Ngal, )
        Base-10 log of Mstar in units of Msun

    redshift: array, shape (Ngal, )

    Returns
    -------
    r50: array, shape (Ngal, )
        size in units of kpc

    z_score: array, shape (Ngal, )
        gaussian random used in MC realization

    """
    logr50_med = _lgr50_kern_bulge(logsm, redshift, bulge_params)
    logr50_med = jnp.clip(logr50_med, jnp.log10(R50_MIN), jnp.log10(R50_MAX))
    z_score = jran.normal(ran_key, shape=logr50_med.shape)
    logr50 = z_score * R50_SCATTER + logr50_med
    r50 = 10**logr50
    return r50, z_score


@jjit
def _lgr50_kern_bulge(lgm, redshift, bulge_params):

    lgm_x0 = _x0_redshift_kern(
        redshift,
        bulge_params.z_x0_x0,
        bulge_params.z_x0_lo,
        bulge_params.z_x0_hi,
    )
    lgm_lgrad_params = (
        bulge_params.lgm_ytp,
        lgm_x0,
        bulge_params.lgm_slope_lo,
        bulge_params.lgm_slope_hi,
    )

    lgr50 = _lgm_lgrad_sig_slope_bulge(lgm, *lgm_lgrad_params)
    return lgr50


@jjit
def _lgm_lgrad_sig_slope_bulge(lgm, ytp, x0, lo, hi):
    return _sig_slope(lgm, XTP, ytp, x0, LGM_LGR_SLOPE_K_BULGE, lo, hi)


@jjit
def _x0_redshift_kern(redshift, z_x0_x0, z_x0_lo, z_x0_hi):
    return _sigmoid(redshift, z_x0_x0, YTP_Z_K_BULGE, z_x0_lo, z_x0_hi)


@jjit
def _slope_redshift_kern(redshift, slope_x0, slope_lo, slope_hi):
    return _sigmoid(redshift, slope_x0, SLOPE_Z_K_BULGE, slope_lo, slope_hi)


_DBS_SIZE_PDICT = dict(
    z_x0_x0=1.0,
    z_x0_lo=10.2,
    z_x0_hi=11.6,
    lgm_ytp=0.2,
    lgm_slope_lo=0.14,
    lgm_slope_hi=0.7,
)
_DBS_BULGE_SIZE_PARAMS = BulgeSizeParams(**_DBS_SIZE_PDICT)
