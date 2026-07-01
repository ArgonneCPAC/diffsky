""""""

from collections import namedtuple

from dsps.sfh import diffburst
from dsps.sfh.diffburst import calc_bursty_age_weights
from dsps.utils import _sigmoid
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ...burstpop import freqburst_mono
from . import ssp_weight_kernels as sspwk

RapidQParams = namedtuple("RapidQParams", ("rq_p_merge_x0", "rq_lg_age_gyr_max"))
DEFAULT_RQ_PARAMS = RapidQParams(rq_p_merge_x0=0.2, rq_lg_age_gyr_max=-1.0)
DEFAULT_RQ_BOUNDS = RapidQParams(
    rq_p_merge_x0=(0.01, 1.0), rq_lg_age_gyr_max=(-5.0, 0.0)
)
K_P_MERGE = 100
K_LG_AGE = 100


calc_bursty_age_weights_vmap = jjit(vmap(calc_bursty_age_weights, in_axes=(0, 0, None)))


@jjit
def get_smooth_ssp_weights_rq(
    t_table,
    sfh_table,
    logsm_obs,
    ssp_data,
    t_obs,
    mzr_params,
    lgmet_scatter,
    p_merge_smooth,
):
    smooth_ssp_weights = sspwk.get_smooth_ssp_weights(
        t_table, sfh_table, logsm_obs, ssp_data, t_obs, mzr_params, lgmet_scatter
    )
    smooth_ssp_weights_rq = modify_smooth_ssp_weights_with_rapid_quenching(
        smooth_ssp_weights, p_merge_smooth, ssp_data
    )
    return smooth_ssp_weights_rq


@jjit
def get_burstiness_rq(
    uran_pburst,
    mc_is_q,
    logsm_obs,
    logssfr_obs,
    age_weights_smooth,
    lgmet_weights,
    ssp_data,
    burstpop_params,
    p_merge_smooth,
):
    n_gals = mc_is_q.shape[0]

    _args = (
        burstpop_params,
        logsm_obs,
        logssfr_obs,
        ssp_data.ssp_lg_age_gyr,
        age_weights_smooth,
    )
    _res = sspwk._calc_bursty_age_weights_vmap(*_args)
    age_weights_bursty, burst_params_bursty = _res

    lgfb_rq = modify_lgfburst_with_rapid_quenching(
        p_merge_smooth, burst_params_bursty.lgfburst
    )
    burst_params_bursty = burst_params_bursty._replace(lgfburst=lgfb_rq)
    age_weights_bursty = calc_bursty_age_weights_vmap(
        burst_params_bursty, age_weights_smooth, ssp_data.ssp_lg_age_gyr
    )

    # Calculate the frequency of SFH bursts
    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        burstpop_params.freqburst_params, logsm_obs, logssfr_obs
    )

    mc_sfh_type = jnp.where(mc_is_q, 0, 1).astype(int)
    msk_bursty = (uran_pburst < p_burst) & (mc_sfh_type == 1)
    mc_sfh_type = jnp.where(msk_bursty, 2, mc_sfh_type).astype(int)

    lgfburst_mc = jnp.where(
        mc_sfh_type < 2, diffburst.LGFBURST_MIN + 0.01, burst_params_bursty.lgfburst
    )
    burst_params_mc = burst_params_bursty._replace(lgfburst=lgfburst_mc)

    age_weights_mc = jnp.where(
        mc_sfh_type.reshape((n_gals, 1)) == 2, age_weights_bursty, age_weights_smooth
    )

    ssp_weights_smooth = sspwk.combine_age_met_weights(
        age_weights_smooth, lgmet_weights
    )
    ssp_weights_bursty = sspwk.combine_age_met_weights(
        age_weights_bursty, lgmet_weights
    )
    ssp_weights_mc = sspwk.combine_age_met_weights(age_weights_mc, lgmet_weights)

    burstiness_info = sspwk.BurstinessInfo(
        ssp_weights_mc,
        ssp_weights_smooth,
        ssp_weights_bursty,
        burst_params_mc,
        burst_params_bursty,
        mc_sfh_type,
        p_burst,
    )
    return burstiness_info


@jjit
def modify_smooth_ssp_weights_with_rapid_quenching(
    ssp_weights_smooth, p_merge, ssp_data
):
    age_weights_rq, __ = get_age_weights_rq_vmap(
        ssp_weights_smooth.age_weights,
        p_merge,
        ssp_data.ssp_lg_age_gyr,
        DEFAULT_RQ_PARAMS,
    )
    ssp_weights_rq = ssp_weights_smooth._replace(age_weights=age_weights_rq)
    return ssp_weights_rq


@jjit
def modify_lgfburst_with_rapid_quenching(p_merge, lgfb_orig):
    lgfb_rq = _sigmoid(
        p_merge,
        DEFAULT_RQ_PARAMS.rq_p_merge_x0,
        K_P_MERGE,
        lgfb_orig,
        diffburst.LGFBURST_MIN + 0.01,
    )
    return lgfb_rq


@jjit
def _rapid_qkern(age_weights, ssp_lg_age_gyr, cutoff_lg_age_gyr):
    """Calculate how rapid quenching changes the stellar age PDF.
    Stars younger than cutoff_lg_age_gyr are removed from the population."""
    ylo = age_weights / 10_000
    yhi = age_weights
    age_weights_rq = _sigmoid(ssp_lg_age_gyr, cutoff_lg_age_gyr, K_LG_AGE, ylo, yhi)
    age_weights_rq = age_weights_rq / age_weights_rq.sum()
    return age_weights_rq


@jjit
def _get_cutoff_lg_age_gyr(p_merge, p_merge_x0, lg_age_gyr_min, lg_age_gyr_max):
    """Calculate the cutoff age in rapid quenching.
    Stars younger than the cutoff age are removed from the population."""
    cutoff_lg_age_gyr = _sigmoid(
        p_merge, p_merge_x0, K_P_MERGE, lg_age_gyr_min, lg_age_gyr_max
    )
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
