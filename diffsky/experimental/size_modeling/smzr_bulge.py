""""""

from collections import namedtuple

from jax import jit as jjit

from ...utils import _sig_slope, _sigmoid

XTP = 10.0
LGM_LGR_SLOPE_K_BULGE = 1.0
YTP_Z_K_BULGE = 2.5
SLOPE_Z_K_BULGE = 5.0


DEFAULT_BULGE_SIZE_PDICT = dict(
    z_x0_x0=1.0,
    z_x0_lo=10.2,
    z_x0_hi=11.6,
    lgm_ytp=0.2,
    lgm_slope_lo=0.14,
    lgm_slope_hi=0.7,
)
BulgeSizeParams = namedtuple("BulgeSizeParams", list(DEFAULT_BULGE_SIZE_PDICT.keys()))
DEFAULT_BULGE_SIZE_PARAMS = BulgeSizeParams(**DEFAULT_BULGE_SIZE_PDICT)


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
