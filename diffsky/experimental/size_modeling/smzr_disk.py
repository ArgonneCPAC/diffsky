""""""

from collections import namedtuple

from jax import jit as jjit

from ...utils import _sig_slope, _sigmoid

XTP = 10.0
LGM_LGR_SLOPE_K_DISK = 0.25
YTP_Z_K_DISK = 2.5
SLOPE_Z_K_DISK = 5.0

DEFAULT_DISK_SIZE_PDICT = dict(
    ytp_x0=1.6,
    ytp_lo=0.63,
    ytp_hi=0.45,
    slope_x0=0.7,
    slope_lo=0.21,
    slope_hi=0.18,
    lgm_x0=11.0,
)
DiskSizeParams = namedtuple("DiskSizeParams", list(DEFAULT_DISK_SIZE_PDICT.keys()))
DEFAULT_DISK_SIZE_PARAMS = DiskSizeParams(**DEFAULT_DISK_SIZE_PDICT)


@jjit
def _lgr50_kern_disk(lgm, redshift, disk_params):
    ytp = _ytp_redshift_kern(
        redshift, disk_params.ytp_x0, disk_params.ytp_lo, disk_params.ytp_hi
    )
    slope_lo = _slope_redshift_kern(
        redshift, disk_params.slope_x0, disk_params.slope_lo, disk_params.slope_hi
    )
    slope_hi = slope_lo
    lgm_lgrad_params = (ytp, disk_params.lgm_x0, slope_lo, slope_hi)

    lgr50 = _lgm_lgrad_sig_slope_disk(lgm, *lgm_lgrad_params)
    return lgr50


@jjit
def _lgm_lgrad_sig_slope_disk(lgm, ytp, x0, lo, hi):
    return _sig_slope(lgm, XTP, ytp, x0, LGM_LGR_SLOPE_K_DISK, lo, hi)


@jjit
def _ytp_redshift_kern(redshift, ytp_x0, ytp_lo, ytp_hi):
    return _sigmoid(redshift, ytp_x0, YTP_Z_K_DISK, ytp_lo, ytp_hi)


@jjit
def _slope_redshift_kern(redshift, slope_x0, slope_lo, slope_hi):
    return _sigmoid(redshift, slope_x0, SLOPE_Z_K_DISK, slope_lo, slope_hi)
