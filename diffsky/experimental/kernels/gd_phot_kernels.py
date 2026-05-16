""""""

from collections import namedtuple
from functools import partial

from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ...dustpop.tw_dust import DEFAULT_DUST_PARAMS
from ...ssp_err_model import ssp_err_model
from .. import mc_diffstarpop_wrappers as mcdw
from .. import photometry_interpolation as photerp
from . import constants, mc_randoms
from . import ssp_weight_kernels as sspwk

LGMET_SCATTER = constants.LGMET_SCATTER


_B = (None, None, 1)
interp_vmap2 = jjit(vmap(jnp.interp, in_axes=_B, out_axes=1))


@partial(jjit, static_argnames=["n_t_table"])
def _mc_phot_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    upid,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssperr_params,
    cosmo_params,
    fb,
    *,
    n_t_table=mcdw.N_T_TABLE,
):
    phot_randoms, diffstarpop_results = mc_randoms.get_phot_randoms(
        ran_key,
        diffstarpop_params,
        mah_params,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        cosmo_params,
    )
    phot_kern_results = _phot_kern(
        phot_randoms,
        diffstarpop_results,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
        n_t_table=n_t_table,
    )
    return phot_kern_results, phot_randoms, diffstarpop_results


@partial(jjit, static_argnames=["n_t_table"])
def _phot_kern(
    phot_randoms,
    diffstarpop_results,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssperr_params,
    cosmo_params,
    fb,
    *,
    n_t_table=mcdw.N_T_TABLE,
):
    """Populate the input lightcone with galaxy SEDs"""

    # Interpolate SSP mag table to z_obs of each galaxy
    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        z_obs, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    # For each filter, calculate λ_eff in the restframe of each galaxy
    wave_eff_galpop = interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    diffstar_info_mc = mcdw.get_diffstar_info(
        mah_params, diffstarpop_results.sfh_params, t_obs, cosmo_params, fb, n_t_table
    )
    diffstar_info_q = mcdw.get_diffstar_info(
        mah_params, diffstarpop_results.sfh_params_q, t_obs, cosmo_params, fb, n_t_table
    )
    diffstar_info_ms = mcdw.get_diffstar_info(
        mah_params,
        diffstarpop_results.sfh_params_ms,
        t_obs,
        cosmo_params,
        fb,
        n_t_table,
    )

    smooth_ssp_weights_mc = sspwk.get_smooth_ssp_weights(
        diffstar_info_mc.t_table,
        diffstar_info_mc.sfh_table,
        diffstar_info_mc.logsm_obs,
        ssp_data,
        t_obs,
        mzr_params,
        LGMET_SCATTER,
    )

    smooth_ssp_weights_q = sspwk.get_smooth_ssp_weights(
        diffstar_info_q.t_table,
        diffstar_info_q.sfh_table,
        diffstar_info_q.logsm_obs,
        ssp_data,
        t_obs,
        mzr_params,
        LGMET_SCATTER,
    )
    smooth_ssp_weights_ms = sspwk.get_smooth_ssp_weights(
        diffstar_info_ms.t_table,
        diffstar_info_ms.sfh_table,
        diffstar_info_ms.logsm_obs,
        ssp_data,
        t_obs,
        mzr_params,
        LGMET_SCATTER,
    )

    burstiness_info_mc = sspwk.get_burstiness(
        phot_randoms.uran_pburst,
        phot_randoms.mc_is_q,
        diffstar_info_mc.logsm_obs,
        diffstar_info_mc.logssfr_obs,
        smooth_ssp_weights_mc.age_weights,
        smooth_ssp_weights_mc.lgmet_weights,
        ssp_data,
        spspop_params.burstpop_params,
    )
    burstiness_info_ms = sspwk.get_burstiness(
        phot_randoms.uran_pburst,
        phot_randoms.mc_is_q,
        diffstar_info_ms.logsm_obs,
        diffstar_info_ms.logssfr_obs,
        smooth_ssp_weights_ms.age_weights,
        smooth_ssp_weights_ms.lgmet_weights,
        ssp_data,
        spspop_params.burstpop_params,
    )
    burstiness_info_q = sspwk.get_burstiness(
        phot_randoms.uran_pburst,
        phot_randoms.mc_is_q,
        diffstar_info_q.logsm_obs,
        diffstar_info_q.logssfr_obs,
        smooth_ssp_weights_q.age_weights,
        smooth_ssp_weights_q.lgmet_weights,
        ssp_data,
        spspop_params.burstpop_params,
    )

    dust_frac_trans_mc, dust_params_mc = sspwk.compute_dust_attenuation(
        phot_randoms.uran_av,
        phot_randoms.uran_delta,
        phot_randoms.uran_funo,
        diffstar_info_mc.logsm_obs,
        diffstar_info_mc.logssfr_obs,
        ssp_data,
        z_obs,
        wave_eff_galpop,
        spspop_params.dustpop_params,
        scatter_params,
    )
    # dust_frac_trans.shape = (n_gals, n_bands, n_age)

    dust_frac_trans_ms, __ = sspwk.compute_dust_attenuation(
        phot_randoms.uran_av,
        phot_randoms.uran_delta,
        phot_randoms.uran_funo,
        diffstar_info_ms.logsm_obs,
        diffstar_info_ms.logssfr_obs,
        ssp_data,
        z_obs,
        wave_eff_galpop,
        spspop_params.dustpop_params,
        scatter_params,
    )
    dust_frac_trans_q, __ = sspwk.compute_dust_attenuation(
        phot_randoms.uran_av,
        phot_randoms.uran_delta,
        phot_randoms.uran_funo,
        diffstar_info_q.logsm_obs,
        diffstar_info_q.logssfr_obs,
        ssp_data,
        z_obs,
        wave_eff_galpop,
        spspop_params.dustpop_params,
        scatter_params,
    )

    # Throw out redundant dust params repeated at each λ_eff
    dust_params_mc = dust_params_mc._replace(
        av=dust_params_mc.av[:, 0, -1],
        delta=dust_params_mc.delta[:, 0],
        funo=dust_params_mc.funo[:, 0],
    )

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors_nonoise_mc = ssp_err_model.frac_ssp_err_at_z_obs_galpop(
        ssperr_params, diffstar_info_mc.logsm_obs, z_obs, wave_eff_galpop
    )
    frac_ssp_errors_mc = ssp_err_model.get_noisy_frac_ssp_errors(
        wave_eff_galpop, frac_ssp_errors_nonoise_mc, phot_randoms.delta_mag_ssp_scatter
    )

    frac_ssp_errors_nonoise_ms = ssp_err_model.frac_ssp_err_at_z_obs_galpop(
        ssperr_params, diffstar_info_ms.logsm_obs, z_obs, wave_eff_galpop
    )
    frac_ssp_errors_ms = ssp_err_model.get_noisy_frac_ssp_errors(
        wave_eff_galpop, frac_ssp_errors_nonoise_ms, phot_randoms.delta_mag_ssp_scatter
    )

    frac_ssp_errors_nonoise_q = ssp_err_model.frac_ssp_err_at_z_obs_galpop(
        ssperr_params, diffstar_info_q.logsm_obs, z_obs, wave_eff_galpop
    )
    frac_ssp_errors_q = ssp_err_model.get_noisy_frac_ssp_errors(
        wave_eff_galpop, frac_ssp_errors_nonoise_q, phot_randoms.delta_mag_ssp_scatter
    )

    obs_mags_mc = sspwk._compute_obs_mags_from_weights(
        diffstar_info_mc.logsm_obs,
        dust_frac_trans_mc,
        frac_ssp_errors_mc,
        ssp_photflux_table,
        burstiness_info_mc.ssp_weights_mc,
    )

    obs_mags_ms = sspwk._compute_obs_mags_from_weights(
        diffstar_info_ms.logsm_obs,
        dust_frac_trans_ms,
        frac_ssp_errors_ms,
        ssp_photflux_table,
        burstiness_info_ms.ssp_weights_smooth,
    )
    obs_mags_q = sspwk._compute_obs_mags_from_weights(
        diffstar_info_q.logsm_obs,
        dust_frac_trans_q,
        frac_ssp_errors_q,
        ssp_photflux_table,
        burstiness_info_q.ssp_weights_smooth,
    )

    obs_mags_bursty = sspwk._compute_obs_mags_from_weights(
        diffstar_info_ms.logsm_obs,
        dust_frac_trans_ms,
        frac_ssp_errors_ms,
        ssp_photflux_table,
        burstiness_info_ms.ssp_weights_bursty,
    )

    n_gals = obs_mags_q.shape[0]
    fq = diffstarpop_results.frac_q
    weights_q = fq
    weights_ms = (1 - fq) * (1 - burstiness_info_ms.p_burst)
    weights_bursty = (1 - fq) * burstiness_info_ms.p_burst

    obs_mags_weighted = (
        (weights_q.reshape((n_gals, 1)) * obs_mags_q)
        + (weights_ms.reshape((n_gals, 1)) * obs_mags_ms)
        + (weights_bursty.reshape((n_gals, 1)) * obs_mags_bursty)
    )

    phot_kern_results = PhotKernResults(
        obs_mags_mc,
        diffstar_info_mc.t_table,
        *diffstarpop_results.sfh_params,
        diffstar_info_mc.sfh_table,
        diffstar_info_mc.logsm_obs,
        diffstar_info_mc.logssfr_obs,
        burstiness_info_mc.mc_sfh_type,
        burstiness_info_mc.ssp_weights_mc,
        smooth_ssp_weights_mc.lgmet_weights,
        *burstiness_info_mc.burst_params_mc,
        *dust_params_mc,
        dust_frac_trans_mc,
        ssp_photflux_table,
        frac_ssp_errors_mc,
        wave_eff_galpop,
        obs_mags_ms,
        obs_mags_q,
        obs_mags_bursty,
        diffstarpop_results.frac_q,
        burstiness_info_ms.p_burst,  # added in
        obs_mags_weighted,
    )
    return phot_kern_results


PHOT_KERN_KEYS = (
    "obs_mags",
    "t_table",
    *DEFAULT_DIFFSTAR_PARAMS._fields,
    "sfh_table",
    "logsm_obs",
    "logssfr_obs",
    "mc_sfh_type",
    "ssp_weights",
    "lgmet_weights",
    *DEFAULT_BURST_PARAMS._fields,
    *DEFAULT_DUST_PARAMS._fields,
    "dust_frac_trans",
    "ssp_photflux_table",
    "frac_ssp_errors",
    "wave_eff_galpop",
    "obs_mags_ms",
    "obs_mags_q",
    "obs_mags_bursty",
    "frac_q",
    "p_burst",  # added in
    "obs_mags_weighted",
)
PhotKernResults = namedtuple("PhotKernResults", PHOT_KERN_KEYS)
