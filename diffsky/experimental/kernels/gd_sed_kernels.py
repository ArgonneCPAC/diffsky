""""""

from collections import namedtuple
from functools import partial

from jax import jit as jjit
from jax import numpy as jnp

from ...merging import merging_model
from ...ssp_err_model import ssp_err_model
from .. import mc_diffstarpop_wrappers as mcdw
from . import rapid_quenching as rq
from . import ssp_weight_kernels as sspwk
from .constants import LGMET_SCATTER


@partial(jjit, static_argnames=["n_t_table"])
def _sed_kern(
    phot_randoms,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    upid,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ssp_data,
    mzr_params,
    spspop_params,
    scatter_params,
    ssperr_params,
    merging_params,
    cosmo_params,
    fb,
    *,
    n_t_table=mcdw.N_T_TABLE,
):
    """"""

    t_table, sfh_table, logsm_obs, logssfr_obs = mcdw.compute_diffstar_info(
        mah_params, sfh_params, t_obs, cosmo_params, fb, n_t_table
    )

    t_infall = t_obs - gyr_since_infall
    logmp_infall = lgmu_infall + logmhost_infall
    p_merge_smooth = merging_model.get_p_merge_from_merging_params(
        merging_params, logmp_infall, logmhost_infall, t_obs, t_infall, upid
    )

    smooth_ssp_weights = rq.get_smooth_ssp_weights_rq(
        t_table,
        sfh_table,
        logsm_obs,
        ssp_data,
        t_obs,
        mzr_params,
        LGMET_SCATTER,
        p_merge_smooth,
    )

    burstiness_info = rq.get_burstiness_rq(
        phot_randoms.uran_pburst,
        phot_randoms.mc_is_q,
        logsm_obs,
        logssfr_obs,
        smooth_ssp_weights.age_weights,
        smooth_ssp_weights.lgmet_weights,
        ssp_data,
        spspop_params.burstpop_params,
        p_merge_smooth,
    )

    # For each filter, calculate λ_eff in the restframe of each galaxy
    n_gals = logsm_obs.shape[0]
    wave_eff_galpop = jnp.tile(ssp_data.ssp_wave, n_gals).reshape((n_gals, -1))

    dust_frac_trans, dust_params = sspwk.compute_dust_attenuation(
        phot_randoms.uran_av,
        phot_randoms.uran_delta,
        phot_randoms.uran_funo,
        logsm_obs,
        logssfr_obs,
        ssp_data,
        z_obs,
        wave_eff_galpop,
        spspop_params.dustpop_params,
        scatter_params,
    )
    dust_params = dust_params._replace(
        av=dust_params.av[:, 0, -1],
        delta=dust_params.delta[:, 0],
        funo=dust_params.funo[:, 0],
    )

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors = ssp_err_model.frac_ssp_err_at_z_obs_galpop(
        ssperr_params, logsm_obs, z_obs, wave_eff_galpop
    )
    frac_ssp_errors = ssp_err_model.get_noisy_frac_ssp_errors(
        wave_eff_galpop, frac_ssp_errors, phot_randoms.delta_mag_ssp_scatter
    )

    n_met, n_age, n_wave = ssp_data.ssp_flux.shape
    dust_frac_trans = dust_frac_trans.swapaxes(1, 2)  # (n_gals, n_age, n_wave)

    a = dust_frac_trans.reshape((n_gals, 1, n_age, n_wave))
    b = frac_ssp_errors.reshape((n_gals, 1, 1, n_wave))
    c = burstiness_info.ssp_weights_mc.reshape((n_gals, n_met, n_age, 1))
    d = ssp_data.ssp_flux.reshape((1, n_met, n_age, n_wave))
    mstar = 10 ** logsm_obs.reshape((n_gals, 1))
    rest_sed = jnp.sum(a * b * c * d, axis=(1, 2)) * mstar

    lgmet_weights = jnp.sum(burstiness_info.ssp_weights_mc, axis=1)

    sed_kern_results = SEDKernResults(
        rest_sed,
        dust_frac_trans,
        frac_ssp_errors,
        burstiness_info.ssp_weights_mc,
        lgmet_weights,
        logsm_obs,
        burstiness_info.burst_params_mc,
        t_table,
        sfh_table,
        p_merge_smooth,
    )
    return sed_kern_results


SEDKernResults = namedtuple(
    "SEDKernResults",
    (
        "rest_sed",
        "dust_frac_trans",
        "frac_ssp_errors",
        "ssp_weights",
        "lgmet_weights",
        "logsm_obs",
        "burst_params",
        "t_table",
        "sfh_table",
        "p_merge_smooth",
    ),
)
