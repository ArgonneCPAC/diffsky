""" """

from collections import OrderedDict, namedtuple

from jax import jit as jjit

from ..utils import _inverse_sigmoid, _sigmoid
from ..utils import tw_utils as twu
from . import halobias_singlez_model as hbszm

HALOBIAS_PDICT = OrderedDict(
    hb_ytp=0.5,
    hb_s0=0.15,
    hb_s1=0.2,
    hb_s2=0.3,
    hb_s3=0.45,
    hb_s4=0.5,
    hb_s5=0.6,
    hb_mz_x0=1.5,
    hb_mz_k=1,
    hb_mz_lo=0.1,
    hb_mz_hi=0.1,
    hb_ytp_z_x0=1.5,
    hb_ytp_z_k=1.0,
    hb_ytp_z_lo=0.1,
    hb_ytp_z_hi=0.1,
)
HaloBiasParams = namedtuple("HaloBiasParams", HALOBIAS_PDICT.keys())
HALOBIAS_PARAMS = HaloBiasParams(**HALOBIAS_PDICT)

_HALOBIAS_UPNAMES = ["u_" + key for key in HALOBIAS_PDICT.keys()]
HaloBiasUParams = namedtuple("HaloBiasUParams", _HALOBIAS_UPNAMES)

HB_YTP_PBOUNDS = (-2, 2)
HB_YTP_X0 = 0.5
BOUNDING_K = 0.1
K_BOUNDS = (0.2, 5.0)


@jjit
def get_bounded_halobias_params(u_hb_params):
    u_hbsz_params = hbszm.HaloBiasUParams._make(
        [getattr(u_hb_params, x) for x in hbszm.HaloBiasUParams._fields]
    )
    hbsz_params = hbszm.get_bounded_halobias_params(u_hbsz_params)

    hb_mz_k = _get_bounded_param_kern(u_hb_params.u_hb_mz_k, *K_BOUNDS)
    hb_ytp_z_k = _get_bounded_param_kern(u_hb_params.u_hb_ytp_z_k, *K_BOUNDS)

    params = (
        *hbsz_params,
        u_hb_params.u_hb_mz_x0,
        hb_mz_k,
        u_hb_params.u_hb_mz_lo,
        u_hb_params.u_hb_mz_hi,
        u_hb_params.u_hb_ytp_z_x0,
        hb_ytp_z_k,
        u_hb_params.u_hb_ytp_z_lo,
        u_hb_params.u_hb_ytp_z_hi,
    )
    params = HaloBiasParams(*params)

    return params


@jjit
def get_unbounded_halobias_params(hb_params):
    hbsz_params = hbszm.HaloBiasParams._make(
        [getattr(hb_params, x) for x in hbszm.HaloBiasParams._fields]
    )
    u_hbsz_params = hbszm.get_unbounded_halobias_params(hbsz_params)

    u_hb_mz_k = _get_unbounded_param_kern(hb_params.hb_mz_k, *K_BOUNDS)
    u_hb_ytp_z_k = _get_unbounded_param_kern(hb_params.hb_ytp_z_k, *K_BOUNDS)

    u_params = (
        *u_hbsz_params,
        hb_params.hb_mz_x0,
        u_hb_mz_k,
        hb_params.hb_mz_lo,
        hb_params.hb_mz_hi,
        hb_params.hb_ytp_z_x0,
        u_hb_ytp_z_k,
        hb_params.hb_ytp_z_lo,
        hb_params.hb_ytp_z_hi,
    )
    u_params = HaloBiasUParams(*u_params)
    return u_params


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
def _get_lgm_at_z_kern(lgm, z, hb_mz_x0, hb_mz_k, hb_mz_lo, hb_mz_hi):
    xtp, ytp = 0.0, 0.0
    return lgm - twu._tw_sig_slope(z, xtp, ytp, hb_mz_x0, hb_mz_k, hb_mz_lo, hb_mz_hi)


@jjit
def _get_hb_ytp_at_z(z, hb_ytp, hb_ytp_z_x0, hb_ytp_z_k, hb_ytp_z_lo, hb_ytp_z_hi):
    xtp, ytp = 0.0, 0.0
    return hb_ytp + twu._tw_sig_slope(
        z, xtp, ytp, hb_ytp_z_x0, hb_ytp_z_k, hb_ytp_z_lo, hb_ytp_z_hi
    )


@jjit
def predict_lgbias_kern(params, lgm, redshift):
    lgbias = _predict_lgbias_kern(
        lgm,
        redshift,
        params.hb_ytp,
        params.hb_s0,
        params.hb_s1,
        params.hb_s2,
        params.hb_s3,
        params.hb_s4,
        params.hb_s5,
        params.hb_mz_x0,
        params.hb_mz_k,
        params.hb_mz_lo,
        params.hb_mz_hi,
        params.hb_ytp_z_x0,
        params.hb_ytp_z_k,
        params.hb_ytp_z_lo,
        params.hb_ytp_z_hi,
    )
    return lgbias


@jjit
def _predict_lgbias_kern(
    lgm,
    redshift,
    hb_ytp,
    hb_s0,
    hb_s1,
    hb_s2,
    hb_s3,
    hb_s4,
    hb_s5,
    hb_mz_x0,
    hb_mz_k,
    hb_mz_lo,
    hb_mz_hi,
    hb_ytp_z_x0,
    hb_ytp_z_k,
    hb_ytp_z_lo,
    hb_ytp_z_hi,
):
    lgm_at_z = _get_lgm_at_z_kern(lgm, redshift, hb_mz_x0, hb_mz_k, hb_mz_lo, hb_mz_hi)
    hb_ytp_at_z = _get_hb_ytp_at_z(
        redshift, hb_ytp, hb_ytp_z_x0, hb_ytp_z_k, hb_ytp_z_lo, hb_ytp_z_hi
    )
    lgbias = hbszm._tw_quintuple_sigmoid_kern(
        lgm_at_z, hb_ytp_at_z, hb_s0, hb_s1, hb_s2, hb_s3, hb_s4, hb_s5
    )
    return lgbias


HALOBIAS_U_PARAMS = get_unbounded_halobias_params(HALOBIAS_PARAMS)
