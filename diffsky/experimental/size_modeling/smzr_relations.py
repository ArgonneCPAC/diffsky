""""""

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp

from ...utils import _sig_slope, _sigmoid

SigmoidParameters = namedtuple("SigmoidParameters", ("x0", "k", "ymin", "ymax"))
LinearParameters = namedtuple("LinearParameters", ("ymin", "slope"))


DiskSizeParameters = namedtuple("DiskSizeParameters", ("a_disk", "alpha_disk"))
DEFAULT_A_DISK = SigmoidParameters(1.33, 2.42, 5.94, 3.82)
DEFAULT_ALPHA_DISK = SigmoidParameters(0.519, 56.4, 0.207, 0.181)
DISK_SIZE_PARAMETERS = DiskSizeParameters(DEFAULT_A_DISK, DEFAULT_ALPHA_DISK)

BulgeSizeParameters = namedtuple(
    "BulgeSizeParameters", ["rp_bulge", "alpha_bulge", "beta_bulge", "logmp_bulge"]
)
DEFAULT_RP_BULGE = LinearParameters(2.93, -0.030)
DEFAULT_ALPHA_BULGE = LinearParameters(0.130, -0.0218)
DEFAULT_BETA_BULGE = LinearParameters(0.71, 0.055)
DEFAULT_LOGMP_BULGE = LinearParameters(10.0, 0.40)

BULGE_SIZE_PARAMETERS = BulgeSizeParameters(
    DEFAULT_RP_BULGE, DEFAULT_ALPHA_BULGE, DEFAULT_BETA_BULGE, DEFAULT_LOGMP_BULGE
)

XTP = 10.0
MR_SLOPE_K = 0.25
YTP_Z_K = 2.5
SLOPE_Z_K = 5.0

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

    lgr50 = _lgm_lgrad_sig_slope(lgm, *lgm_lgrad_params)
    return lgr50


@jjit
def _lgm_lgrad_sig_slope(lgm, ytp, x0, lo, hi):
    return _sig_slope(lgm, XTP, ytp, x0, MR_SLOPE_K, lo, hi)


@jjit
def _ytp_redshift_kern(redshift, ytp_x0, ytp_lo, ytp_hi):
    return _sigmoid(redshift, ytp_x0, YTP_Z_K, ytp_lo, ytp_hi)


@jjit
def _slope_redshift_kern(redshift, slope_x0, slope_lo, slope_hi):
    return _sigmoid(redshift, slope_x0, SLOPE_Z_K, slope_lo, slope_hi)


@jjit
def _linear_kern(x, y0, slope, *, x0=0.0):
    return y0 + slope * (x - x0)


@jjit
def _r50_med_vs_mstar_powlaw_kern(mstar, a, alpha, m0=5e10):
    """Formerly median_r50_vs_mstar"""
    r50_med = a * jnp.power(mstar / m0, alpha)
    return r50_med


@jjit
def _r50_med_vs_mstar_double_powlaw_kern(mstar, rp, alpha, beta, logmp, delta=6):
    """Formerly median_r50_vs_mstar2"""
    mp = jnp.power(10, logmp)
    term1 = rp * jnp.power(mstar / mp, alpha)
    term2 = 0.5 * jnp.power((1 + jnp.power(mstar / mp, delta)), (beta - alpha) / delta)
    r50_med = term1 * term2
    return r50_med


@jjit
def _get_parameter_zevolution_disk(redshift, disk_size_params):
    evolved_a_disk = _sigmoid(redshift, *disk_size_params.a_disk)
    evolved_alpha_disk = _sigmoid(redshift, *disk_size_params.alpha_disk)
    disk_size_params = DiskSizeParameters(evolved_a_disk, evolved_alpha_disk)
    return disk_size_params


@jjit
def _get_parameter_zevolution_bulge(redshift, bulge_size_params):
    evolved_rp_bulge = _linear_kern(redshift, *bulge_size_params.rp_bulge)
    evolved_alpha_bulge = _linear_kern(redshift, *bulge_size_params.alpha_bulge)
    evolved_beta_bulge = _linear_kern(redshift, *bulge_size_params.beta_bulge)
    evolved_logmp_bulge = _linear_kern(redshift, *bulge_size_params.logmp_bulge)
    bulge_size_params = BulgeSizeParameters(
        evolved_rp_bulge, evolved_alpha_bulge, evolved_beta_bulge, evolved_logmp_bulge
    )
    return bulge_size_params


@jjit
def _disk_median_r50(mstar, redshift, disk_size_params):
    disk_size_params_at_z = _get_parameter_zevolution_disk(redshift, disk_size_params)
    r50 = _r50_med_vs_mstar_powlaw_kern(mstar, *disk_size_params_at_z)
    return r50


@jjit
def _bulge_median_r50(mstar, redshift, bulge_size_params):
    par_values = _get_parameter_zevolution_bulge(redshift, bulge_size_params)
    r50 = _r50_med_vs_mstar_double_powlaw_kern(mstar, *par_values)
    return r50
