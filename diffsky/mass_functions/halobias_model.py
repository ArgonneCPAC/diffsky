""" """

from collections import OrderedDict, namedtuple

from jax import jit as jjit

from ..utils import tw_utils as twu
from . import halobias_singlez_model as hbsm

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
)
HaloBiasParams = namedtuple("HaloBiasParams", HALOBIAS_PDICT.keys())
HALOBIAS_PARAMS = HaloBiasParams(**HALOBIAS_PDICT)

_HALOBIAS_UPNAMES = ["u_" + key for key in HALOBIAS_PDICT.keys()]
HaloBiasUParams = namedtuple("HaloBiasUParams", _HALOBIAS_UPNAMES)

HB_YTP_PBOUNDS = (-2, 2)
HB_YTP_X0 = 0.5
BOUNDING_K = 0.1


def get_bounded_halobias_params():
    raise NotImplementedError()


@jjit
def _get_lgm_at_z_kern(lgm, z, hb_mz_x0, hb_mz_k, hb_mz_lo, hb_mz_hi):
    xtp, ytp = 0.0, 0.0
    return lgm - twu._tw_sig_slope(z, xtp, ytp, hb_mz_x0, hb_mz_k, hb_mz_lo, hb_mz_hi)


@jjit
def _get_hb_ytp_at_z(
    hb_ytp_at_z0, z, hb_ytp_z_x0, hb_ytp_z_k, hb_ytp_z_lo, hb_ytp_z_hi
):
    xtp, ytp = 0.0, 0.0
    return hb_ytp_at_z0 + twu._tw_sig_slope(
        z, xtp, ytp, hb_ytp_z_x0, hb_ytp_z_k, hb_ytp_z_lo, hb_ytp_z_hi
    )


def predict_lgbias_kern(params, lgm, z):
    raise NotImplementedError()


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
        hb_ytp, redshift, hb_ytp_z_x0, hb_ytp_z_k, hb_ytp_z_lo, hb_ytp_z_hi
    )
    lgbias = hbsm._tw_quintuple_sigmoid_kern(
        lgm_at_z, hb_ytp_at_z, hb_s0, hb_s1, hb_s2, hb_s3, hb_s4, hb_s5
    )
    return lgbias
