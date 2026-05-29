""""""

from collections import namedtuple
from dsps.utils import _sigmoid
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from . import ssp_weight_kernels as sspwk

RapidQParams = namedtuple("RapidQParams", ("rq_p_merge_x0", "rq_lg_age_gyr_max"))
DEFAULT_RQ_PARAMS = RapidQParams(rq_p_merge_x0=0.2, rq_lg_age_gyr_max=-1.0)
DEFAULT_RQ_BOUNDS = RapidQParams(
    rq_p_merge_x0=(0.01, 1.0), rq_lg_age_gyr_max=(-5.0, 0.0)
)


@jjit
def modify_burstiness_info_with_rapid_quenching(
    burstiness_info, p_merge, ssp_data, lgmet_weights
):
    age_weights_mc = jnp.sum(burstiness_info.ssp_weights_mc, axis=1)
    age_weights_mc_rq, __ = get_age_weights_rq_vmap(
        age_weights_mc, p_merge, ssp_data.ssp_lg_age_gyr, DEFAULT_RQ_PARAMS
    )
    ssp_weights_mc_rq = sspwk.combine_age_met_weights(age_weights_mc_rq, lgmet_weights)

    age_weights_smooth = jnp.sum(burstiness_info.ssp_weights_smooth, axis=1)
    age_weights_smooth_rq, __ = get_age_weights_rq_vmap(
        age_weights_smooth, p_merge, ssp_data.ssp_lg_age_gyr, DEFAULT_RQ_PARAMS
    )
    ssp_weights_smooth_rq = sspwk.combine_age_met_weights(
        age_weights_smooth_rq, lgmet_weights
    )

    age_weights_bursty = jnp.sum(burstiness_info.ssp_weights_bursty, axis=1)
    age_weights_bursty_rq, __ = get_age_weights_rq_vmap(
        age_weights_bursty, p_merge, ssp_data.ssp_lg_age_gyr, DEFAULT_RQ_PARAMS
    )
    ssp_weights_bursty_rq = sspwk.combine_age_met_weights(
        age_weights_bursty_rq, lgmet_weights
    )

    burstiness_info_rq = burstiness_info._replace(
        ssp_weights_mc=ssp_weights_mc_rq,
        ssp_weights_smooth=ssp_weights_smooth_rq,
        ssp_weights_bursty=ssp_weights_bursty_rq,
    )

    return burstiness_info_rq


@jjit
def _rapid_qkern(age_weights, ssp_lg_age_gyr, cutoff_lg_age_gyr):
    """Calculate how rapid quenching changes the stellar age PDF.
    Stars younger than cutoff_lg_age_gyr are removed from the population."""
    k = 100.0
    ylo = age_weights / 10_000
    yhi = age_weights
    age_weights_rq = _sigmoid(ssp_lg_age_gyr, cutoff_lg_age_gyr, k, ylo, yhi)
    age_weights_rq = age_weights_rq / age_weights_rq.sum()
    return age_weights_rq


@jjit
def _get_cutoff_lg_age_gyr(p_merge, p_merge_x0, lg_age_gyr_min, lg_age_gyr_max):
    """Calculate the cutoff age in rapid quenching.
    Stars younger than the cutoff age are removed from the population."""
    k = 100.0
    cutoff_lg_age_gyr = _sigmoid(p_merge, p_merge_x0, k, lg_age_gyr_min, lg_age_gyr_max)
    return cutoff_lg_age_gyr


@jjit
def _age_weights_rapid_q(age_weights, p_merge, ssp_lg_age_gyr, rapidq_params):
    """Calculate rapid quenching vs p_merge"""
    lg_age_gyr_min = ssp_lg_age_gyr[0] - 1  # ensure no effect for low-p_merge limit
    cutoff_lg_age_gyr = _get_cutoff_lg_age_gyr(
        p_merge,
        rapidq_params.rq_p_merge_x0,
        lg_age_gyr_min,
        rapidq_params.rq_lg_age_gyr_max,
    )
    age_weights_rq = _rapid_qkern(age_weights, ssp_lg_age_gyr, cutoff_lg_age_gyr)
    return age_weights_rq, cutoff_lg_age_gyr


get_age_weights_rq_vmap = vmap(_age_weights_rapid_q, in_axes=(0, 0, None, None))
