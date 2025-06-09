# flake8: noqa: E402
""" """

from jax import config

config.update("jax_enable_x64", True)

from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.metallicity import umzr
from dsps.sed import metallicity_weights as zmetw
from dsps.sed import stellar_age_weights as saw
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .. import mc_diffsky as mcd
from ..burstpop import diffqburstpop_mono
from ..dustpop import tw_dustpop_mono, tw_dustpop_mono_noise
from ..phot_utils import get_wave_eff_from_tcurves, load_interpolated_lsst_curves
from ..ssp_err_model import ssp_err_model
from . import precompute_ssp_phot as psp
from .scatter import DEFAULT_SCATTER_PARAMS

# gal_t_table, gal_sfr_table, ssp_lg_age_gyr, t_obs, sfr_min
_A = (None, 0, None, None, None)
_calc_age_weights_galpop = jjit(vmap(saw.calc_age_weights_from_sfh_table, in_axes=_A))

# gal_lgmet, gal_lgmet_scatter, ssp_lgmet
_M = (0, None, None)
_calc_lgmet_weights_galpop = jjit(
    vmap(zmetw.calc_lgmet_weights_from_lognormal_mdf, in_axes=_M)
)

# diffburstpop_params, logsm, logssfr, ssp_lg_age_gyr, smooth_age_weights
_B = (None, 0, 0, None, 0)
_calc_bursty_age_weights_vmap = jjit(
    vmap(
        diffqburstpop_mono.calc_bursty_age_weights_from_diffburstpop_params, in_axes=_B
    )
)

Z_KCORRECT = 0.1


def mc_diffsky_galpop_lsst_phot(
    ran_key,
    z_obs,
    lgmp_min,
    volume_com,
    ssp_data,
    cosmo_params=DEFAULT_COSMOLOGY,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    dustpop_params=tw_dustpop_mono.DEFAULT_DUSTPOP_PARAMS,
    mzr_params=umzr.DEFAULT_MZR_PARAMS,
    lgmet_scatter=umzr.MZR_SCATTER,
    diffburstpop_params=diffqburstpop_mono.DEFAULT_DIFFBURSTPOP_PARAMS,
    scatter_params=DEFAULT_SCATTER_PARAMS,
    ssp_err_pop_params=ssp_err_model.DEFAULT_SSPERR_PARAMS,
    n_t=mcd.N_T,
    drn_ssp_data=mcd.DSPS_DATA_DRN,
    return_internal_quantities=False,
):
    diffstar_key, ran_key = jran.split(ran_key, 2)
    diffstar_data = mcd.mc_diffstar_galpop(
        diffstar_key,
        z_obs,
        lgmp_min,
        volume_com=volume_com,
        cosmo_params=cosmo_params,
        diffstarpop_params=diffstarpop_params,
        n_t=n_t,
        return_internal_quantities=return_internal_quantities,
    )
    diffsky_data = predict_lsst_phot_from_diffstar(
        diffstar_data,
        ran_key,
        z_obs,
        ssp_data,
        cosmo_params=cosmo_params,
        dustpop_params=dustpop_params,
        mzr_params=mzr_params,
        lgmet_scatter=lgmet_scatter,
        diffburstpop_params=diffburstpop_params,
        scatter_params=scatter_params,
        ssp_err_pop_params=ssp_err_pop_params,
        drn_ssp_data=drn_ssp_data,
        return_internal_quantities=return_internal_quantities,
    )
    return diffsky_data


def mc_diffsky_cenpop_lsst_phot(
    ran_key,
    z_obs,
    hosts_logmh_at_z,
    ssp_data,
    cosmo_params=DEFAULT_COSMOLOGY,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    dustpop_params=tw_dustpop_mono.DEFAULT_DUSTPOP_PARAMS,
    mzr_params=umzr.DEFAULT_MZR_PARAMS,
    lgmet_scatter=umzr.MZR_SCATTER,
    diffburstpop_params=diffqburstpop_mono.DEFAULT_DIFFBURSTPOP_PARAMS,
    scatter_params=DEFAULT_SCATTER_PARAMS,
    ssp_err_pop_params=ssp_err_model.DEFAULT_SSPERR_PARAMS,
    n_t=mcd.N_T,
    drn_ssp_data=mcd.DSPS_DATA_DRN,
    return_internal_quantities=False,
):
    diffstar_key, ran_key = jran.split(ran_key, 2)
    diffstar_data = mcd.mc_diffstar_cenpop(
        diffstar_key,
        z_obs,
        hosts_logmh_at_z=hosts_logmh_at_z,
        cosmo_params=cosmo_params,
        diffstarpop_params=diffstarpop_params,
        n_t=n_t,
        return_internal_quantities=return_internal_quantities,
    )
    diffsky_data = predict_lsst_phot_from_diffstar(
        diffstar_data,
        ran_key,
        z_obs,
        ssp_data,
        cosmo_params=cosmo_params,
        dustpop_params=dustpop_params,
        mzr_params=mzr_params,
        lgmet_scatter=lgmet_scatter,
        diffburstpop_params=diffburstpop_params,
        scatter_params=scatter_params,
        ssp_err_pop_params=ssp_err_pop_params,
        drn_ssp_data=drn_ssp_data,
        return_internal_quantities=return_internal_quantities,
    )
    return diffsky_data


def predict_lsst_phot_from_diffstar(
    diffstar_data,
    ran_key,
    z_obs,
    ssp_data,
    cosmo_params=DEFAULT_COSMOLOGY,
    dustpop_params=tw_dustpop_mono.DEFAULT_DUSTPOP_PARAMS,
    mzr_params=umzr.DEFAULT_MZR_PARAMS,
    lgmet_scatter=umzr.MZR_SCATTER,
    diffburstpop_params=diffqburstpop_mono.DEFAULT_DIFFBURSTPOP_PARAMS,
    scatter_params=DEFAULT_SCATTER_PARAMS,
    ssp_err_pop_params=ssp_err_model.DEFAULT_SSPERR_PARAMS,
    drn_ssp_data=mcd.DSPS_DATA_DRN,
    return_internal_quantities=False,
    z_kcorrect=Z_KCORRECT,
):
    diffsky_data = diffstar_data.copy()

    # Compute age weights and metallicity weights
    lgmet_key, ran_key = jran.split(ran_key, 2)
    lgmet_med = umzr.mzr_model(
        diffsky_data["logsm_obs"], diffsky_data["t_obs"], *mzr_params
    )
    diffsky_data["lgmet_nonoise"] = lgmet_med
    unorm = jran.normal(lgmet_key, shape=lgmet_med.shape) * lgmet_scatter
    diffsky_data["lgmet_med"] = lgmet_med + unorm

    diffsky_data["smooth_age_weights"] = _calc_age_weights_galpop(
        diffsky_data["t_table"],
        diffsky_data["sfh"],
        ssp_data.ssp_lg_age_gyr,
        diffsky_data["t_obs"],
        saw.SFR_MIN,
    )

    if return_internal_quantities:
        diffsky_data["smooth_age_weights_ms"] = _calc_age_weights_galpop(
            diffsky_data["t_table"],
            diffsky_data["sfh_ms"],
            ssp_data.ssp_lg_age_gyr,
            diffsky_data["t_obs"],
            saw.SFR_MIN,
        )
        diffsky_data["smooth_age_weights_q"] = _calc_age_weights_galpop(
            diffsky_data["t_table"],
            diffsky_data["sfh_q"],
            ssp_data.ssp_lg_age_gyr,
            diffsky_data["t_obs"],
            saw.SFR_MIN,
        )

    diffsky_data["lgmet_weights"] = _calc_lgmet_weights_galpop(
        diffsky_data["lgmet_med"], lgmet_scatter, ssp_data.ssp_lgmet
    )

    # Compute age weights with burstiness
    _args = (
        diffburstpop_params,
        diffsky_data["logsm_obs"],
        diffsky_data["logssfr_obs"],
        ssp_data.ssp_lg_age_gyr,
        diffsky_data["smooth_age_weights"],
    )
    bursty_age_weights, burst_params = _calc_bursty_age_weights_vmap(*_args)
    diffsky_data["bursty_age_weights"] = bursty_age_weights
    for param, pname in zip(burst_params, burst_params._fields):
        diffsky_data[pname] = param

    if return_internal_quantities:
        _args = (
            diffburstpop_params,
            diffsky_data["logsm_obs"],
            diffsky_data["logssfr_obs"],
            ssp_data.ssp_lg_age_gyr,
            diffsky_data["smooth_age_weights_ms"],
        )
        bursty_age_weights, burst_params = _calc_bursty_age_weights_vmap(*_args)
        diffsky_data["bursty_age_weights_ms"] = bursty_age_weights

        _args = (
            diffburstpop_params,
            diffsky_data["logsm_obs"],
            diffsky_data["logssfr_obs"],
            ssp_data.ssp_lg_age_gyr,
            diffsky_data["smooth_age_weights_q"],
        )
        bursty_age_weights, burst_params = _calc_bursty_age_weights_vmap(*_args)
        diffsky_data["bursty_age_weights_q"] = bursty_age_weights

    n_gals = diffsky_data["sfh"].shape[0]
    n_met, n_age = ssp_data.ssp_flux.shape[0:2]

    _wmet = diffsky_data["lgmet_weights"].reshape((n_gals, n_met, 1))

    _amet = diffsky_data["smooth_age_weights"].reshape((n_gals, 1, n_age))
    smooth_weights = _wmet * _amet
    _norm = jnp.sum(smooth_weights, axis=(1, 2))
    smooth_weights = smooth_weights / _norm.reshape((n_gals, 1, 1))
    diffsky_data["smooth_ssp_weights"] = smooth_weights

    if return_internal_quantities:
        _amet = diffsky_data["smooth_age_weights_ms"].reshape((n_gals, 1, n_age))
        smooth_weights = _wmet * _amet
        _norm = jnp.sum(smooth_weights, axis=(1, 2))
        smooth_weights = smooth_weights / _norm.reshape((n_gals, 1, 1))
        diffsky_data["smooth_ssp_weights_ms"] = smooth_weights

        _amet = diffsky_data["smooth_age_weights_q"].reshape((n_gals, 1, n_age))
        smooth_weights = _wmet * _amet
        _norm = jnp.sum(smooth_weights, axis=(1, 2))
        smooth_weights = smooth_weights / _norm.reshape((n_gals, 1, 1))
        diffsky_data["smooth_ssp_weights_q"] = smooth_weights

    _bmet = diffsky_data["bursty_age_weights"].reshape((n_gals, 1, n_age))
    bursty_weights = _wmet * _bmet
    _norm = jnp.sum(bursty_weights, axis=(1, 2))
    bursty_weights = bursty_weights / _norm.reshape((n_gals, 1, 1))
    diffsky_data["bursty_ssp_weights"] = bursty_weights

    if return_internal_quantities:
        _bmet = diffsky_data["bursty_age_weights_ms"].reshape((n_gals, 1, n_age))
        bursty_weights = _wmet * _bmet
        _norm = jnp.sum(bursty_weights, axis=(1, 2))
        bursty_weights = bursty_weights / _norm.reshape((n_gals, 1, 1))
        diffsky_data["bursty_ssp_weights_ms"] = bursty_weights

        _bmet = diffsky_data["bursty_age_weights_q"].reshape((n_gals, 1, n_age))
        bursty_weights = _wmet * _bmet
        _norm = jnp.sum(bursty_weights, axis=(1, 2))
        bursty_weights = bursty_weights / _norm.reshape((n_gals, 1, 1))
        diffsky_data["bursty_ssp_weights_q"] = bursty_weights

    # Compute dust attenuation with and without noise in dust params

    lsst_tcurves_interp, lsst_tcurves_sparse = load_interpolated_lsst_curves(
        ssp_data.ssp_wave, drn_ssp_data=drn_ssp_data
    )
    rest_wave_eff_ugrizy_aa = get_wave_eff_from_tcurves(lsst_tcurves_sparse, z_obs)
    obs_wave_eff_ugrizy_aa = get_wave_eff_from_tcurves(lsst_tcurves_sparse, z_obs)
    # n_bands = rest_wave_eff_ugrizy_aa.size

    ssp_flux_table_multiband = psp.get_ssp_restflux_table(
        ssp_data, lsst_tcurves_sparse, z_kcorrect
    )
    ssp_obs_flux_table_multiband = psp.get_ssp_obsflux_table(
        ssp_data, lsst_tcurves_sparse, z_obs, cosmo_params
    )

    av_key, delta_key, funo_key, ran_key = jran.split(ran_key, 4)
    uran_av = jran.uniform(av_key, shape=(n_gals,))
    uran_delta = jran.uniform(delta_key, shape=(n_gals,))
    uran_funo = jran.uniform(funo_key, shape=(n_gals,))

    # Transmission fraction in rest frame (with and without noise)
    _res = calc_dust_ftrans_vmap(
        dustpop_params,
        rest_wave_eff_ugrizy_aa,
        diffsky_data["logsm_obs"],
        diffsky_data["logssfr_obs"],
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    nonoise_ftrans_rest, noisy_ftrans_rest, dust_params_rest, noisy_dust_params_rest = (
        _res
    )
    # Transmission fraction in observed frame (with and without noise)
    _res = calc_dust_ftrans_vmap(
        dustpop_params,
        obs_wave_eff_ugrizy_aa,
        diffsky_data["logsm_obs"],
        diffsky_data["logssfr_obs"],
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    nonoise_ftrans_obs, noisy_ftrans_obs, dust_params_obs, noisy_dust_params_obs = _res
    for param, pname in zip(dust_params_obs, dust_params_obs._fields):
        diffsky_data[pname + "_nonoise"] = param
    for param, pname in zip(noisy_dust_params_obs, noisy_dust_params_obs._fields):
        diffsky_data[pname] = param

    # Precomputed SSP fluxes in rest and observer frame
    logsm_obs = diffsky_data["logsm_obs"].reshape((n_gals, 1, 1, 1))
    gal_flux_table_nodust = ssp_flux_table_multiband * 10**logsm_obs
    gal_obs_flux_table_nodust = ssp_obs_flux_table_multiband * 10**logsm_obs

    # Restframe flux table with dust
    n_gals, n_filters, n_met, n_age = gal_flux_table_nodust.shape
    _s = (n_gals, n_filters, 1, n_age)
    gal_flux_table_dust = gal_flux_table_nodust * noisy_ftrans_rest.reshape(_s)

    # Observer frame flux table with dust
    gal_obs_flux_table_dust = gal_obs_flux_table_nodust * noisy_ftrans_obs.reshape(_s)

    # Rest and observed flux: smooth SFH & no dust
    w = diffsky_data["smooth_ssp_weights"].reshape((n_gals, 1, n_met, n_age))

    flux = jnp.sum(gal_flux_table_nodust * w, axis=(2, 3))
    mag_smooth_nodust = -2.5 * jnp.log10(flux)
    obs_flux = jnp.sum(gal_obs_flux_table_nodust * w, axis=(2, 3))
    obs_mag_smooth_nodust = -2.5 * jnp.log10(obs_flux)

    # Rest and observed flux: smooth SFH & with dust
    flux = jnp.sum(gal_flux_table_dust * w, axis=(2, 3))
    mag_smooth_dust = -2.5 * jnp.log10(flux)
    obs_flux = jnp.sum(gal_obs_flux_table_dust * w, axis=(2, 3))
    obs_mag_smooth_dust = -2.5 * jnp.log10(obs_flux)

    # Rest and observed flux: bursty SFH & no dust
    w = diffsky_data["bursty_ssp_weights"].reshape((n_gals, 1, n_met, n_age))

    flux = jnp.sum(gal_flux_table_nodust * w, axis=(2, 3))
    mag_bursty_nodust = -2.5 * jnp.log10(flux)
    obs_flux = jnp.sum(gal_obs_flux_table_nodust * w, axis=(2, 3))
    obs_mag_bursty_nodust = -2.5 * jnp.log10(obs_flux)

    # Rest and observed flux: bursty SFH & with dust
    flux = jnp.sum(gal_flux_table_dust * w, axis=(2, 3))
    mag_bursty_dust = -2.5 * jnp.log10(flux)
    obs_flux = jnp.sum(gal_obs_flux_table_dust * w, axis=(2, 3))
    obs_mag_bursty_dust = -2.5 * jnp.log10(obs_flux)

    diffsky_data["rest_ugrizy_smooth_nodust"] = mag_smooth_nodust
    diffsky_data["rest_ugrizy_bursty_nodust"] = mag_bursty_nodust
    diffsky_data["rest_ugrizy_smooth_dust"] = mag_smooth_dust
    diffsky_data["rest_ugrizy_bursty_dust"] = mag_bursty_dust
    diffsky_data["obs_ugrizy_smooth_nodust"] = obs_mag_smooth_nodust
    diffsky_data["obs_ugrizy_bursty_nodust"] = obs_mag_bursty_nodust
    diffsky_data["obs_ugrizy_smooth_dust"] = obs_mag_smooth_dust
    diffsky_data["obs_ugrizy_bursty_dust"] = obs_mag_bursty_dust

    ssp_ran_key, ran_key = jran.split(ran_key, 2)

    # Delta mags
    delta_mag_obs = ssp_err_model.noisy_delta_mag(
        ssp_err_pop_params,
        z_obs,
        diffsky_data["logsm_obs"],
        obs_wave_eff_ugrizy_aa,
        ssp_err_model.LAMBDA_REST,
        ssp_ran_key,
    )

    delta_mag_rest = ssp_err_model.noisy_delta_mag(
        ssp_err_pop_params,
        z_obs,
        diffsky_data["logsm_obs"],
        rest_wave_eff_ugrizy_aa,
        ssp_err_model.LAMBDA_REST,
        ssp_ran_key,
    )

    diffsky_data["rest_ugrizy_smooth_nodust_ssperrs"] = (
        mag_smooth_nodust + delta_mag_rest
    )
    diffsky_data["rest_ugrizy_bursty_nodust_ssperrs"] = (
        mag_bursty_nodust + delta_mag_rest
    )
    diffsky_data["rest_ugrizy_smooth_dust_ssperrs"] = mag_smooth_dust + delta_mag_rest
    diffsky_data["rest_ugrizy_bursty_dust_ssperrs"] = mag_bursty_dust + delta_mag_rest
    diffsky_data["obs_ugrizy_smooth_nodust_ssperrs"] = (
        obs_mag_smooth_nodust + delta_mag_obs
    )
    diffsky_data["obs_ugrizy_bursty_nodust_ssperrs"] = (
        obs_mag_bursty_nodust + delta_mag_obs
    )
    diffsky_data["obs_ugrizy_smooth_dust_ssperrs"] = obs_mag_smooth_dust + delta_mag_obs
    diffsky_data["obs_ugrizy_bursty_dust_ssperrs"] = obs_mag_bursty_dust + delta_mag_obs

    if return_internal_quantities:
        # Main sequence
        # Rest and observed flux: smooth SFH with and without dust
        w = diffsky_data["smooth_ssp_weights_ms"].reshape((n_gals, 1, n_met, n_age))

        mags_nodust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_nodust * w, axis=(2, 3)))
        mags_dust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_dust * w, axis=(2, 3)))
        diffsky_data["rest_ugrizy_smooth_nodust_ms"] = mags_nodust
        diffsky_data["rest_ugrizy_smooth_dust_ms"] = mags_dust
        diffsky_data["rest_ugrizy_smooth_nodust_ms_ssperrs"] = (
            mags_nodust + delta_mag_rest
        )
        diffsky_data["rest_ugrizy_smooth_dust_ms_ssperrs"] = mags_dust + delta_mag_rest

        obs_mags_nodust = -2.5 * jnp.log10(
            jnp.sum(gal_obs_flux_table_nodust * w, axis=(2, 3))
        )
        obs_mags_dust = -2.5 * jnp.log10(
            jnp.sum(gal_obs_flux_table_dust * w, axis=(2, 3))
        )
        diffsky_data["obs_ugrizy_smooth_nodust_ms"] = obs_mags_nodust
        diffsky_data["obs_ugrizy_smooth_dust_ms"] = obs_mags_dust
        diffsky_data["obs_ugrizy_smooth_nodust_ms_ssperrs"] = (
            obs_mags_nodust + delta_mag_obs
        )
        diffsky_data["obs_ugrizy_smooth_dust_ms_ssperrs"] = (
            obs_mags_dust + delta_mag_obs
        )

        # Quenched sequence
        # Rest and observed flux: smooth SFH with and without dust
        w = diffsky_data["smooth_ssp_weights_q"].reshape((n_gals, 1, n_met, n_age))

        mags_nodust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_nodust * w, axis=(2, 3)))
        mags_dust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_dust * w, axis=(2, 3)))
        diffsky_data["rest_ugrizy_smooth_nodust_q"] = mags_nodust
        diffsky_data["rest_ugrizy_smooth_dust_q"] = mags_dust
        diffsky_data["rest_ugrizy_smooth_nodust_q_ssperrs"] = (
            mags_nodust + delta_mag_rest
        )
        diffsky_data["rest_ugrizy_smooth_dust_q_ssperrs"] = mags_dust + delta_mag_rest
        obs_mags_nodust = -2.5 * jnp.log10(
            jnp.sum(gal_obs_flux_table_nodust * w, axis=(2, 3))
        )
        obs_mags_dust = -2.5 * jnp.log10(
            jnp.sum(gal_obs_flux_table_dust * w, axis=(2, 3))
        )
        diffsky_data["obs_ugrizy_smooth_nodust_q"] = obs_mags_nodust
        diffsky_data["obs_ugrizy_smooth_dust_q"] = obs_mags_dust
        diffsky_data["obs_ugrizy_smooth_nodust_q_ssperrs"] = (
            obs_mags_nodust + delta_mag_obs
        )
        diffsky_data["obs_ugrizy_smooth_dust_q_ssperrs"] = obs_mags_dust + delta_mag_obs

        # Main sequence
        # Rest and observed flux: bursty SFH with and without dust
        w = diffsky_data["bursty_ssp_weights_ms"].reshape((n_gals, 1, n_met, n_age))

        mags_nodust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_nodust * w, axis=(2, 3)))
        mags_dust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_dust * w, axis=(2, 3)))
        diffsky_data["rest_ugrizy_bursty_nodust_ms"] = mags_nodust
        diffsky_data["rest_ugrizy_bursty_dust_ms"] = mags_dust
        diffsky_data["rest_ugrizy_bursty_nodust_ms_ssperrs"] = (
            mags_nodust + delta_mag_rest
        )
        diffsky_data["rest_ugrizy_bursty_dust_ms_ssperrs"] = mags_dust + delta_mag_rest
        obs_mags_nodust = -2.5 * jnp.log10(
            jnp.sum(gal_obs_flux_table_nodust * w, axis=(2, 3))
        )
        obs_mags_dust = -2.5 * jnp.log10(
            jnp.sum(gal_obs_flux_table_dust * w, axis=(2, 3))
        )
        diffsky_data["obs_ugrizy_bursty_nodust_ms"] = obs_mags_nodust
        diffsky_data["obs_ugrizy_bursty_dust_ms"] = obs_mags_dust
        diffsky_data["obs_ugrizy_bursty_nodust_ms_ssperrs"] = (
            obs_mags_nodust + delta_mag_obs
        )
        diffsky_data["obs_ugrizy_bursty_dust_ms_ssperrs"] = (
            obs_mags_dust + delta_mag_obs
        )

        # Quenched sequence
        # Rest and observed flux: bursty SFH with and without dust
        w = diffsky_data["bursty_ssp_weights_q"].reshape((n_gals, 1, n_met, n_age))

        mags_nodust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_nodust * w, axis=(2, 3)))
        mags_dust = -2.5 * jnp.log10(jnp.sum(gal_flux_table_dust * w, axis=(2, 3)))
        diffsky_data["rest_ugrizy_bursty_nodust_q"] = mags_nodust
        diffsky_data["rest_ugrizy_bursty_dust_q"] = mags_dust
        diffsky_data["rest_ugrizy_bursty_nodust_q_ssperrs"] = (
            mags_nodust + delta_mag_rest
        )
        diffsky_data["rest_ugrizy_bursty_dust_q_ssperrs"] = mags_dust + delta_mag_rest

        obs_mags_nodust = -2.5 * jnp.log10(
            jnp.sum(gal_obs_flux_table_nodust * w, axis=(2, 3))
        )
        obs_mags_dust = -2.5 * jnp.log10(
            jnp.sum(gal_obs_flux_table_dust * w, axis=(2, 3))
        )
        diffsky_data["obs_ugrizy_bursty_nodust_q"] = obs_mags_nodust
        diffsky_data["obs_ugrizy_bursty_dust_q"] = obs_mags_dust
        diffsky_data["obs_ugrizy_bursty_nodust_q_ssperrs"] = (
            obs_mags_nodust + delta_mag_obs
        )
        diffsky_data["obs_ugrizy_bursty_dust_q_ssperrs"] = obs_mags_dust + delta_mag_obs

        # Store additional internal quantities
        diffsky_data["frac_trans_nonoise_rest"] = nonoise_ftrans_rest
        diffsky_data["frac_trans_noisy_rest"] = noisy_ftrans_rest
        diffsky_data["rest_wave_eff_ugrizy_aa"] = rest_wave_eff_ugrizy_aa
        diffsky_data["frac_trans_nonoise_obs"] = nonoise_ftrans_obs
        diffsky_data["frac_trans_noisy_obs"] = noisy_ftrans_obs
        diffsky_data["obs_wave_eff_ugrizy_aa"] = obs_wave_eff_ugrizy_aa

        diffsky_data["ssp_flux_table_multiband"] = ssp_flux_table_multiband
        diffsky_data["ssp_obs_flux_table_multiband"] = ssp_obs_flux_table_multiband

    return diffsky_data


_A = (None, 0, None, None, None, None, None, None, None, None)
_B = [None, None, 0, 0, None, None, 0, 0, 0, None]

_f = jjit(
    vmap(
        tw_dustpop_mono_noise.calc_ftrans_singlegal_singlewave_from_dustpop_params,
        in_axes=_A,
    )
)
calc_dust_ftrans_vmap = jjit(vmap(_f, in_axes=_B))
