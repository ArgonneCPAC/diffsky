""" """

from jax import jit as jjit

from ..utils import tw_utils as twu
from . import halobias_singlez_model as hbsm


@jjit
def _get_lgm_at_z_kern(lgm, z, x0, k, lo, hi):
    xtp, ytp = 0.0, 0.0
    return twu._tw_sig_slope(z, xtp, ytp, x0, k, lo, hi)


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
    hb_z_z,
    hb_z_x0,
    hb_z_k,
    hb_z_lo,
    hb_z_hi,
):
    lgm_at_z = _get_lgm_at_z_kern(
        lgm, redshift, hb_z_z, hb_z_x0, hb_z_k, hb_z_lo, hb_z_hi
    )
    lgbias = hbsm._tw_quintuple_sigmoid_kern(
        lgm_at_z, hb_ytp, hb_s0, hb_s1, hb_s2, hb_s3, hb_s4, hb_s5
    )
    return lgbias
