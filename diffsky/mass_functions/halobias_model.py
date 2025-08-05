""" """

from collections import OrderedDict, namedtuple

from jax import jit as jjit

from ..utils import _inverse_sigmoid, _sigmoid
from ..utils import tw_utils as twu
from . import halobias_singlez_model as hbszm

HALOBIAS_PDICT = OrderedDict(
    hb_ytp=-0.07,
    hb_s0=0.30,
    hb_s1=0.38,
    hb_s2=0.45,
    hb_s3=0.47,
    hb_s4=0.49,
    hb_s5=0.50,
    hb_ytp_z_x0=-0.56,
    hb_ytp_z_k=0.92,
    hb_ytp_z_lo=-0.23,
    hb_ytp_z_hi=0.21,
    hb_x0_lgm_us=13.01,
    hb_k_lgm_us=1.20,
    hb_ylo_lgm_us_zlo=0.50,
    hb_yhi_lgm_us_zlo=-0.18,
    hb_ylo_lgm_us_zhi=0.07,
    hb_yhi_lgm_us_zhi=-0.10,
    hb_dus_z_x0=-1.09,
    hb_dus_z_k=1.55,
)
HaloBiasParams = namedtuple("HaloBiasParams", HALOBIAS_PDICT.keys())
HALOBIAS_PARAMS = HaloBiasParams(**HALOBIAS_PDICT)

_HALOBIAS_UPNAMES = ["u_" + key for key in HALOBIAS_PDICT.keys()]
HaloBiasUParams = namedtuple("HaloBiasUParams", _HALOBIAS_UPNAMES)

HB_YTP_PBOUNDS = (-2, 2)
HB_YTP_X0 = 0.5
BOUNDING_K = 0.1
K_BOUNDS = (0.2, 5.0)
DUS_LGM_XTP = 10.0


@jjit
def get_unbounded_halobias_params(hb_params):
    hbsz_params = hbszm.HaloBiasParams._make(
        [getattr(hb_params, x) for x in hbszm.HaloBiasParams._fields]
    )
    u_hbsz_params = hbszm.get_unbounded_halobias_params(hbsz_params)

    u_hb_ytp_z_k = _get_unbounded_param_kern(hb_params.hb_ytp_z_k, *K_BOUNDS)
    u_hb_k_lgm_us = _get_unbounded_param_kern(hb_params.hb_k_lgm_us, *K_BOUNDS)
    u_hb_dus_z_k = _get_unbounded_param_kern(hb_params.hb_dus_z_k, *K_BOUNDS)

    u_params = (
        *u_hbsz_params,
        hb_params.hb_ytp_z_x0,
        u_hb_ytp_z_k,
        hb_params.hb_ytp_z_lo,
        hb_params.hb_ytp_z_hi,
        hb_params.hb_x0_lgm_us,
        u_hb_k_lgm_us,
        hb_params.hb_ylo_lgm_us_zlo,
        hb_params.hb_yhi_lgm_us_zlo,
        hb_params.hb_ylo_lgm_us_zhi,
        hb_params.hb_yhi_lgm_us_zhi,
        hb_params.hb_dus_z_x0,
        u_hb_dus_z_k,
    )
    u_params = HaloBiasUParams(*u_params)
    return u_params


@jjit
def get_bounded_halobias_params(u_hb_params):
    u_hbsz_params = hbszm.HaloBiasUParams._make(
        [getattr(u_hb_params, x) for x in hbszm.HaloBiasUParams._fields]
    )
    hbsz_params = hbszm.get_bounded_halobias_params(u_hbsz_params)

    hb_ytp_z_k = _get_bounded_param_kern(u_hb_params.u_hb_ytp_z_k, *K_BOUNDS)
    hb_k_lgm_us = _get_bounded_param_kern(u_hb_params.u_hb_k_lgm_us, *K_BOUNDS)
    hb_dus_z_k = _get_bounded_param_kern(u_hb_params.u_hb_dus_z_k, *K_BOUNDS)

    params = (
        *hbsz_params,
        u_hb_params.u_hb_ytp_z_x0,
        hb_ytp_z_k,
        u_hb_params.u_hb_ytp_z_lo,
        u_hb_params.u_hb_ytp_z_hi,
        u_hb_params.u_hb_x0_lgm_us,
        hb_k_lgm_us,
        u_hb_params.u_hb_ylo_lgm_us_zlo,
        u_hb_params.u_hb_yhi_lgm_us_zlo,
        u_hb_params.u_hb_ylo_lgm_us_zhi,
        u_hb_params.u_hb_yhi_lgm_us_zhi,
        u_hb_params.u_hb_dus_z_x0,
        hb_dus_z_k,
    )

    params = HaloBiasParams(*params)

    return params


@jjit
def _get_bounded_param_kern(u_p, lo, hi):
    x0 = 0.5 * (lo + hi)
    p = _sigmoid(u_p, x0, BOUNDING_K, lo, hi)
    return p


@jjit
def _get_unbounded_param_kern(p, lo, hi):
    x0 = 0.5 * (lo + hi)
    u_p = _inverse_sigmoid(p, x0, BOUNDING_K, lo, hi)
    return u_p


@jjit
def _delta_u_slope_kern(
    lgm,
    z,
    hb_x0_lgm_us,
    hb_k_lgm_us,
    hb_ylo_lgm_us_zlo,
    hb_yhi_lgm_us_zlo,
    hb_ylo_lgm_us_zhi,
    hb_yhi_lgm_us_zhi,
    hb_dus_z_x0,
    hb_dus_z_k,
):
    dus_z_zlo = _sigmoid(
        lgm, hb_x0_lgm_us, hb_k_lgm_us, hb_ylo_lgm_us_zlo, hb_yhi_lgm_us_zlo
    )
    dus_z_zhi = _sigmoid(
        lgm, hb_x0_lgm_us, hb_k_lgm_us, hb_ylo_lgm_us_zhi, hb_yhi_lgm_us_zhi
    )
    delta_u_slope = twu._tw_sig_slope(
        z, DUS_LGM_XTP, 0.0, hb_dus_z_x0, hb_dus_z_k, dus_z_zlo, dus_z_zhi
    )
    return delta_u_slope


@jjit
def _get_hb_ytp_at_z(z, hb_ytp, hb_ytp_z_x0, hb_ytp_z_k, hb_ytp_z_lo, hb_ytp_z_hi):
    xtp, ytp = 0.0, 0.0
    return hb_ytp + twu._tw_sig_slope(
        z, xtp, ytp, hb_ytp_z_x0, hb_ytp_z_k, hb_ytp_z_lo, hb_ytp_z_hi
    )


@jjit
def _get_hbszm_params_at_hiz(hbm_params, z):
    hbszm_params = hbszm.HaloBiasParams(
        hbm_params.hb_ytp,
        hbm_params.hb_s0,
        hbm_params.hb_s1,
        hbm_params.hb_s2,
        hbm_params.hb_s3,
        hbm_params.hb_s4,
        hbm_params.hb_s5,
    )

    hbszm_u_params = hbszm.get_unbounded_halobias_params(hbszm_params)

    delta_u_slope = _delta_u_slope_kern(
        hbszm.LGM_TABLE,
        z,
        hbm_params.hb_x0_lgm_us,
        hbm_params.hb_k_lgm_us,
        hbm_params.hb_ylo_lgm_us_zlo,
        hbm_params.hb_yhi_lgm_us_zlo,
        hbm_params.hb_ylo_lgm_us_zhi,
        hbm_params.hb_yhi_lgm_us_zhi,
        hbm_params.hb_dus_z_x0,
        hbm_params.hb_dus_z_k,
    )
    slope_u_params_at_z = [
        u_s + d_us for (u_s, d_us) in zip(hbszm_u_params[1:], delta_u_slope)
    ]
    hbszm_u_params = hbszm.HaloBiasUParams(
        hbszm_u_params.u_hb_ytp, *slope_u_params_at_z
    )
    hbszm_params = hbszm.get_bounded_halobias_params(hbszm_u_params)

    hb_ytp_at_z = _get_hb_ytp_at_z(
        z,
        hbm_params.hb_ytp,
        hbm_params.hb_ytp_z_x0,
        hbm_params.hb_ytp_z_k,
        hbm_params.hb_ytp_z_lo,
        hbm_params.hb_ytp_z_hi,
    )
    hbszm_params = hbszm_params._replace(hb_ytp=hb_ytp_at_z)
    return hbszm_params


@jjit
def predict_lgbias_kern(hbm_params, lgm, redshift):
    hbszm_params = _get_hbszm_params_at_hiz(hbm_params, redshift)
    lgbias = hbszm._tw_quintuple_sigmoid_kern(lgm, *hbszm_params)
    return lgbias


HALOBIAS_U_PARAMS = get_unbounded_halobias_params(HALOBIAS_PARAMS)
