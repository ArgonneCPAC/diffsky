# flake8: noqa: E402
""""""

from jax import config

config.update("jax_enable_x64", True)

from functools import partial

from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ..param_utils import diffsky_param_wrapper as dpw
from ..ssp_err_model2 import ssp_err_model
from . import mc_diffstarpop_wrappers as mcdw
from .kernels import mc_phot_kernels as mcpk
from .kernels.ssp_weight_kernels_repro import (
    compute_burstiness,
    compute_dust_attenuation,
    get_smooth_ssp_weights,
)


def mc_dbk_phot(
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
):
    phot_key, dbk_key = jran.split(ran_key, 2)
    phot_kern_results, phot_randoms = mcpk._mc_phot_kern(
        phot_key,
        diffstarpop_params,
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
        ssp_err_pop_params,
        cosmo_params,
        fb,
    )

    burst_params = DEFAULT_BURST_PARAMS._replace(
        lgfburst=phot_kern_results.lgfburst,
        lgyr_peak=phot_kern_results.lgyr_peak,
        lgyr_max=phot_kern_results.lgyr_max,
    )
    _ret2 = mcpk._mc_dbk_kern(
        t_obs,
        ssp_data,
        phot_kern_results.t_table,
        phot_kern_results.sfh_table,
        burst_params,
        phot_kern_results.lgmet_weights,
        dbk_key,
    )
    dbk_randoms, dbk_weights, disk_bulge_history = _ret2

    # For each filter, calculate λ_eff in the restframe of each galaxy
    wave_eff_galpop = mcpk.interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    _ret3 = _get_dbk_phot_from_dbk_weights(
        phot_kern_results.ssp_photflux_table,
        dbk_weights,
        phot_kern_results.dust_frac_trans,
        wave_eff_galpop,
        phot_kern_results.frac_ssp_errors,
        phot_randoms.delta_mag_ssp_scatter,
    )
    obs_mags_bulge, obs_mags_disk, obs_mags_knots = _ret3

    dbk_phot_info = mcpk.MCDBKPhotInfo(
        **phot_kern_results._asdict(),
        **phot_randoms._asdict(),
        **dbk_randoms._asdict(),
        **disk_bulge_history.fbulge_params._asdict(),
        bulge_to_total_history=disk_bulge_history.bulge_to_total_history,
        obs_mags_bulge=obs_mags_bulge,
        obs_mags_disk=obs_mags_disk,
        obs_mags_knots=obs_mags_knots,
    )
    return dbk_phot_info


@jjit
def _get_dbk_phot_from_dbk_weights(
    ssp_photflux_table,
    dbk_weights,
    dust_frac_trans,
    wave_eff_galpop,
    frac_ssp_err_nonoise,
    delta_mag_ssp_scatter,
):
    n_gals, n_bands, n_met, n_age = ssp_photflux_table.shape

    # Reshape arrays before calculating galaxy magnitudes
    _ftrans = dust_frac_trans.reshape((n_gals, n_bands, 1, n_age))

    # Calculate fractional changes to SSP fluxes
    # frac_ssp_err_noise = frac_ssp_errors * 10 ** (-0.4 * delta_scatter)
    frac_ssp_err_noise = ssp_err_model.get_noisy_frac_ssp_errors(
        wave_eff_galpop, frac_ssp_err_nonoise, delta_mag_ssp_scatter
    )

    _ferr_ssp = frac_ssp_err_noise.reshape((n_gals, n_bands, 1, 1))

    _w_bulge = dbk_weights.ssp_weights_bulge.reshape((n_gals, 1, n_met, n_age))
    _w_dd = dbk_weights.ssp_weights_disk.reshape((n_gals, 1, n_met, n_age))
    _w_knot = dbk_weights.ssp_weights_knots.reshape((n_gals, 1, n_met, n_age))

    _mstar_bulge = dbk_weights.mstar_bulge.reshape((n_gals, 1))
    _mstar_disk = dbk_weights.mstar_disk.reshape((n_gals, 1))
    _mstar_knots = dbk_weights.mstar_knots.reshape((n_gals, 1))

    integrand_bulge = ssp_photflux_table * _w_bulge * _ftrans * _ferr_ssp
    flux_bulge = jnp.sum(integrand_bulge, axis=(2, 3)) * _mstar_bulge
    obs_mags_bulge = -2.5 * jnp.log10(flux_bulge)

    integrand_disk = ssp_photflux_table * _w_dd * _ftrans * _ferr_ssp
    flux_disk = jnp.sum(integrand_disk, axis=(2, 3)) * _mstar_disk
    obs_mags_disk = -2.5 * jnp.log10(flux_disk)

    integrand_knots = ssp_photflux_table * _w_knot * _ftrans * _ferr_ssp
    flux_knots = jnp.sum(integrand_knots, axis=(2, 3)) * _mstar_knots
    obs_mags_knots = -2.5 * jnp.log10(flux_knots)

    return obs_mags_bulge, obs_mags_disk, obs_mags_knots


@partial(jjit, static_argnames=["n_t_table"])
def _sed_kern(
    phot_randoms,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
    n_t_table=mcdw.N_T_TABLE,
):
    """"""

    t_table, sfh_table, logsm_obs, logssfr_obs = mcdw.compute_diffstar_info(
        mah_params, sfh_params, t_obs, cosmo_params, fb, n_t_table
    )

    age_weights_smooth, lgmet_weights = get_smooth_ssp_weights(
        t_table, sfh_table, logsm_obs, ssp_data, t_obs, mzr_params, mcpk.LGMET_SCATTER
    )

    _res = compute_burstiness(
        phot_randoms.uran_pburst,
        phot_randoms.mc_is_q,
        logsm_obs,
        logssfr_obs,
        age_weights_smooth,
        lgmet_weights,
        ssp_data,
        spspop_params.burstpop_params,
    )
    ssp_weights, burst_params, mc_sfh_type = _res

    # For each filter, calculate λ_eff in the restframe of each galaxy
    n_gals = logsm_obs.shape[0]
    wave_eff_galpop = jnp.tile(ssp_data.ssp_wave, n_gals).reshape((n_gals, -1))

    dust_frac_trans, dust_params = compute_dust_attenuation(
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

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_errors = ssp_err_model.frac_ssp_err_at_z_obs_galpop(
        ssp_err_pop_params, logsm_obs, z_obs, wave_eff_galpop
    )

    n_met, n_age, n_wave = ssp_data.ssp_flux.shape
    dust_frac_trans = dust_frac_trans.swapaxes(1, 2)  # (n_gals, n_age, n_wave)

    a = dust_frac_trans.reshape((n_gals, 1, n_age, n_wave))
    b = frac_ssp_errors.reshape((n_gals, 1, 1, n_wave))
    c = ssp_weights.reshape((n_gals, n_met, n_age, 1))
    d = ssp_data.ssp_flux.reshape((1, n_met, n_age, n_wave))
    mstar = 10 ** logsm_obs.reshape((n_gals, 1))
    rest_sed = jnp.sum(a * b * c * d, axis=(1, 2)) * mstar

    sed_kern_results = (rest_sed, dust_frac_trans, frac_ssp_errors, ssp_weights)
    return sed_kern_results


def mcw_lc_phot(
    ran_key,
    lc_data,
    diffstarpop_params=dpw.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
    mzr_params=dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
    spspop_params=dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
    scatter_params=dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
    ssperr_params=dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    phot_kern_results, phot_randoms = mcpk._mc_phot_kern(
        ran_key,
        diffstarpop_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
    )
    phot_kern_results = phot_kern_results._asdict()
    for key, val in zip(lc_data.mah_params._fields, lc_data.mah_params):
        phot_kern_results[key] = val
    return phot_kern_results


def mcw_lc_sed(
    ran_key,
    lc_data,
    diffstarpop_params=dpw.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
    mzr_params=dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
    spspop_params=dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
    scatter_params=dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
    ssperr_params=dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    phot_kern_results, phot_randoms = mcpk._mc_phot_kern(
        ran_key,
        diffstarpop_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [getattr(phot_kern_results, key) for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    sed_kern_results = _sed_kern(
        phot_randoms,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        mzr_params,
        spspop_params,
        scatter_params,
        ssperr_params,
        cosmo_params,
        fb,
    )
    rest_sed = sed_kern_results[0]
    phot_kern_results = phot_kern_results._asdict()
    phot_kern_results["rest_sed"] = rest_sed
    return phot_kern_results
