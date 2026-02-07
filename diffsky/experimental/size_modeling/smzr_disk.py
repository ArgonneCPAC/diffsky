""""""

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ...utils import _sig_slope, _sigmoid

R50_MIN, R50_MAX = 0.1, 40.0
R50_SCATTER = 0.2


XTP = 10.0
LGM_LGR_SLOPE_K_DISK = 0.25
YTP_Z_K_DISK = 2.5
SLOPE_Z_K_DISK = 5.0

DEFAULT_DISK_SIZE_PDICT = dict(
    ytp_x0=1.6,
    ytp_lo=0.73,
    ytp_hi=0.55,
    slope_x0=0.7,
    slope_lo=0.21,
    slope_hi=0.18,
    lgm_x0=11.0,
)
DiskSizeParams = namedtuple("DiskSizeParams", list(DEFAULT_DISK_SIZE_PDICT.keys()))
DEFAULT_DISK_SIZE_PARAMS = DiskSizeParams(**DEFAULT_DISK_SIZE_PDICT)


def mc_r50_disk_size(logsm, redshift, ran_key, disk_params=DEFAULT_DISK_SIZE_PARAMS):
    """Monte Carlo realization of size--mass relation for disks

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
    logr50_med = _lgr50_kern_disk(logsm, redshift, disk_params)
    logr50_med = jnp.clip(logr50_med, jnp.log10(R50_MIN), jnp.log10(R50_MAX))
    z_score = jran.normal(ran_key, shape=logr50_med.shape)
    logr50 = z_score * R50_SCATTER + logr50_med
    r50 = 10**logr50
    return r50, z_score


@jjit
def _lgr50_kern_disk(logsm, redshift, disk_params):
    ytp = _ytp_redshift_kern(
        redshift, disk_params.ytp_x0, disk_params.ytp_lo, disk_params.ytp_hi
    )
    slope_lo = _slope_redshift_kern(
        redshift, disk_params.slope_x0, disk_params.slope_lo, disk_params.slope_hi
    )
    slope_hi = slope_lo
    lgm_lgrad_params = (ytp, disk_params.lgm_x0, slope_lo, slope_hi)

    lgr50 = _lgm_lgrad_sig_slope_disk(logsm, *lgm_lgrad_params)
    return lgr50


@jjit
def _lgm_lgrad_sig_slope_disk(logsm, ytp, x0, lo, hi):
    return _sig_slope(logsm, XTP, ytp, x0, LGM_LGR_SLOPE_K_DISK, lo, hi)


@jjit
def _ytp_redshift_kern(redshift, ytp_x0, ytp_lo, ytp_hi):
    return _sigmoid(redshift, ytp_x0, YTP_Z_K_DISK, ytp_lo, ytp_hi)


@jjit
def _slope_redshift_kern(redshift, slope_x0, slope_lo, slope_hi):
    return _sigmoid(redshift, slope_x0, SLOPE_Z_K_DISK, slope_lo, slope_hi)


_DBS_SIZE_PDICT = dict(
    ytp_x0=1.6,
    ytp_lo=0.63,
    ytp_hi=0.45,
    slope_x0=0.7,
    slope_lo=0.21,
    slope_hi=0.18,
    lgm_x0=11.0,
)
_DBS_DISK_SIZE_PARAMS = DiskSizeParams(**_DBS_SIZE_PDICT)
