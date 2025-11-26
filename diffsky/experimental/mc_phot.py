""""""

from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple

from diffstar.diffstarpop.param_utils import mc_select_diffstar_params
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ..ssp_err_model import ssp_err_model
from . import lc_phot_kern
from . import mc_diffstarpop_wrappers as mcdw
from . import photometry_interpolation as photerp
from .kernels.ssp_weight_kernels import (
    compute_burstiness,
    compute_dust_attenuation,
    compute_frac_ssp_errors,
    compute_obs_mags_ms_q,
    get_smooth_ssp_weights,
)
from .mc_diffstarpop_wrappers import N_T_TABLE

LGMET_SCATTER = 0.2


@jjit
def _mc_phot_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
    n_t_table=N_T_TABLE,
):
    """Populate the input lightcone with galaxy SEDs"""
    n_z_table, n_bands, n_met, n_age = precomputed_ssp_mag_table.shape
    n_gals = z_obs.size

    # Calculate SFH with diffstarpop
    ran_key, sfh_key = jran.split(ran_key, 2)
    args = (diffstarpop_params, sfh_key, mah_params, t_obs, cosmo_params, fb)
    diffstar_galpop = mcdw.diffstarpop_cen_wrapper(*args, n_t_table=n_t_table)

    smooth_ssp_weights = get_smooth_ssp_weights(
        diffstar_galpop, ssp_data, t_obs, mzr_params, LGMET_SCATTER
    )

    burstiness = compute_burstiness(
        diffstar_galpop, smooth_ssp_weights, ssp_data, spspop_params.burstpop_params
    )

    # Interpolate SSP mag table to z_obs of each galaxy
    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        z_obs, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    # For each filter, calculate λ_eff in the restframe of each galaxy
    wave_eff_galpop = lc_phot_kern.interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    # Generate randoms for stochasticity in dust attenuation curves
    ran_key, dust_key = jran.split(ran_key, 2)
    dust_att = compute_dust_attenuation(
        dust_key,
        diffstar_galpop,
        ssp_data,
        z_obs,
        wave_eff_galpop,
        spspop_params.dustpop_params,
        scatter_params,
    )

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors = compute_frac_ssp_errors(
        ssp_err_pop_params, z_obs, diffstar_galpop, wave_eff_galpop
    )
    ran_key, ssp_q_key, ssp_ms_key = jran.split(ran_key, 3)
    delta_scatter_ms = ssp_err_model.compute_delta_scatter(
        ssp_ms_key, frac_ssp_errors.ms
    )
    delta_scatter_q = ssp_err_model.compute_delta_scatter(ssp_q_key, frac_ssp_errors.q)

    _obs_mags = compute_obs_mags_ms_q(
        diffstar_galpop,
        dust_att,
        frac_ssp_errors,
        ssp_photflux_table,
        smooth_ssp_weights.weights.ms,
        burstiness.weights.ms,
        smooth_ssp_weights.weights.q,
        delta_scatter_ms,
        delta_scatter_q,
    )
    obs_mags_q, obs_mags_smooth_ms, obs_mags_bursty_ms = _obs_mags

    weights_smooth_ms = (1 - diffstar_galpop.frac_q) * (1 - burstiness.p_burst)
    weights_bursty_ms = (1 - diffstar_galpop.frac_q) * burstiness.p_burst
    weights_q = diffstar_galpop.frac_q

    # Generate Monte Carlo noise to stochastically select q, or ms-smooth, or ms-bursty
    ran_key, smooth_sfh_key = jran.split(ran_key, 2)
    uran_smooth_sfh = jran.uniform(smooth_sfh_key, shape=(n_gals,))

    # Calculate CDFs from weights
    # 0 < cdf < f_q ==> quenched
    # f_q < cdf < f_q + f_smooth_ms ==> smooth main sequence
    # f_q + f_smooth_ms < cdf < 1 ==> bursty main sequence
    cdf_q = weights_q
    cdf_ms = weights_q + weights_smooth_ms
    mc_q = uran_smooth_sfh < cdf_q
    diffstar_params = mc_select_diffstar_params(
        diffstar_galpop.diffstar_params_q, diffstar_galpop.diffstar_params_ms, mc_q
    )

    # mc_sfh_type = 0 for quenched, 1 for smooth ms, 2 for bursty ms
    mc_sfh_type = jnp.zeros(n_gals).astype(int)
    mc_smooth_ms = (uran_smooth_sfh >= cdf_q) & (uran_smooth_sfh < cdf_ms)
    mc_sfh_type = jnp.where(mc_smooth_ms, 1, mc_sfh_type)
    mc_bursty_ms = uran_smooth_sfh >= cdf_ms
    mc_sfh_type = jnp.where(mc_bursty_ms, 2, mc_sfh_type)
    sfh_table = jnp.where(
        mc_sfh_type.reshape((n_gals, 1)) == 0,
        diffstar_galpop.sfh_q,
        diffstar_galpop.sfh_ms,
    )

    # Calculate stochastic realization of SSP weights
    mc_smooth_ms = mc_smooth_ms.reshape((n_gals, 1, 1))
    mc_bursty_ms = mc_bursty_ms.reshape((n_gals, 1, 1))
    ssp_weights = jnp.copy(smooth_ssp_weights.weights.q)
    ssp_weights = jnp.where(mc_smooth_ms, smooth_ssp_weights.weights.ms, ssp_weights)
    ssp_weights = jnp.where(mc_bursty_ms, burstiness.weights.ms, ssp_weights)
    # ssp_weights.shape = (n_gals, n_met, n_age)

    # Reshape stellar mass used to normalize SED
    logsm_obs = jnp.where(
        mc_sfh_type > 0, diffstar_galpop.logsm_obs_ms, diffstar_galpop.logsm_obs_q
    )

    # Calculate specific SFR at z_obs
    logssfr_obs = jnp.where(
        mc_sfh_type > 0, diffstar_galpop.logssfr_obs_ms, diffstar_galpop.logssfr_obs_q
    )

    # Reshape boolean array storing SFH type
    msk_ms = mc_sfh_type.reshape((-1, 1)) == 1
    msk_bursty = mc_sfh_type.reshape((-1, 1)) == 2

    # Select observed mags according to SFH selection
    obs_mags = jnp.copy(obs_mags_q)
    obs_mags = jnp.where(msk_ms, obs_mags_smooth_ms, obs_mags)
    obs_mags = jnp.where(msk_bursty, obs_mags_bursty_ms, obs_mags)

    msk_q = mc_sfh_type == 0
    av = jnp.where(
        msk_q.reshape((n_gals, 1)),
        dust_att.dust_params.q.av[:, 0, :],
        dust_att.dust_params.ms.av[:, 0, :],
    )
    delta = jnp.where(
        msk_q, dust_att.dust_params.q.delta[:, 0], dust_att.dust_params.ms.delta[:, 0]
    )
    funo = jnp.where(
        msk_q, dust_att.dust_params.q.funo[:, 0], dust_att.dust_params.ms.funo[:, 0]
    )
    dust_params = dust_att.dust_params.q._make((av, delta, funo))

    phot_info = PhotInfo(
        logmp_obs=diffstar_galpop.logmp_obs,
        logsm_obs=logsm_obs,
        logssfr_obs=logssfr_obs,
        sfh_table=sfh_table,
        obs_mags=obs_mags,
        diffstar_params=diffstar_params,
        mc_sfh_type=mc_sfh_type,
        burst_params=burstiness.burst_params,
        dust_params=dust_params,
        ssp_weights=ssp_weights,
        uran_av=dust_att.dust_scatter.av,
        uran_delta=dust_att.dust_scatter.delta,
        uran_funo=dust_att.dust_scatter.funo,
        logsm_obs_ms=diffstar_galpop.logsm_obs_ms,
        logsm_obs_q=diffstar_galpop.logsm_obs_q,
        logssfr_obs_ms=diffstar_galpop.logssfr_obs_ms,
        logssfr_obs_q=diffstar_galpop.logssfr_obs_q,
        delta_scatter_ms=delta_scatter_ms,
        delta_scatter_q=delta_scatter_q,
    )

    return phot_info._asdict()


PHOT_INFO_KEYS = (
    "logmp_obs",
    "logsm_obs",
    "logssfr_obs",
    "sfh_table",
    "obs_mags",
    "diffstar_params",
    "mc_sfh_type",
    "burst_params",
    "dust_params",
    "ssp_weights",
    "uran_av",
    "uran_delta",
    "uran_funo",
    "logsm_obs_ms",
    "logssfr_obs_ms",
    "logsm_obs_q",
    "logssfr_obs_q",
    "delta_scatter_ms",
    "delta_scatter_q",
)
PhotInfo = namedtuple("PhotInfo", PHOT_INFO_KEYS)
