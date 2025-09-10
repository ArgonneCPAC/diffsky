# flake8: noqa: E402
""" """
from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple

from diffstarpop.param_utils import mc_select_diffstar_params
from dsps.metallicity import umzr
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..burstpop import freqburst_mono
from ..param_utils import diffsky_param_wrapper as dpw
from ..ssp_err_model import ssp_err_model
from . import lc_phot_kern
from . import mc_lightcone_halos as mclh
from . import photometry_interpolation as photerp

SED_INFO_KEYS = (
    "diffstar_params",
    "burst_params",
    "ssp_weights",
    "wave_eff_galpop",
    "frac_ssp_err_sed_ms",
    "frac_ssp_err_sed_q",
    "ftrans_sed_ms",
    "ftrans_sed_q",
)
SedInfo = namedtuple("SedInfo", SED_INFO_KEYS)
SEDINFO_EMPTY = SedInfo._make([None] * len(SedInfo._fields))

ssp_err_interp = jjit(vmap(ssp_err_model._tw_wave_interp_kern, in_axes=(None, 0, 0)))


def mc_diffsky_seds(u_param_arr, ran_key, lc_data):
    u_param_collection = dpw.get_u_param_collection_from_u_param_array(u_param_arr)
    param_collection = dpw.get_param_collection_from_u_param_collection(
        *u_param_collection
    )
    sed_data = mc_diffsky_seds_kern(ran_key, *lc_data[1:], *param_collection)
    return sed_data


def mc_diffsky_seds_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
    logmp0,
    t_table,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
):
    n_z_table, n_bands, n_met, n_age = precomputed_ssp_mag_table.shape
    n_gals = logmp0.size

    ran_key, sfh_key = jran.split(ran_key, 2)
    diffstar_galpop = lc_phot_kern.diffstarpop_lc_cen_wrapper(
        diffstarpop_params, sfh_key, mah_params, logmp0, t_table, t_obs
    )

    smooth_age_weights_ms = lc_phot_kern.calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_ms, ssp_data.ssp_lg_age_gyr, t_obs
    )
    smooth_age_weights_q = lc_phot_kern.calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_q, ssp_data.ssp_lg_age_gyr, t_obs
    )

    _args = (
        spspop_params.burstpop_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
        ssp_data.ssp_lg_age_gyr,
        smooth_age_weights_ms,
    )
    bursty_age_weights_ms, burst_params = lc_phot_kern._calc_bursty_age_weights_vmap(
        *_args
    )

    p_burst_ms = freqburst_mono.get_freqburst_from_freqburst_params(
        spspop_params.burstpop_params.freqburst_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
    )

    lgmet_med_ms = umzr.mzr_model(diffstar_galpop.logsm_obs_ms, t_obs, *mzr_params)
    lgmet_med_q = umzr.mzr_model(diffstar_galpop.logsm_obs_q, t_obs, *mzr_params)

    lgmet_weights_ms = lc_phot_kern._calc_lgmet_weights_galpop(
        lgmet_med_ms, lc_phot_kern.LGMET_SCATTER, ssp_data.ssp_lgmet
    )
    lgmet_weights_q = lc_phot_kern._calc_lgmet_weights_galpop(
        lgmet_med_q, lc_phot_kern.LGMET_SCATTER, ssp_data.ssp_lgmet
    )

    _w_age_q = smooth_age_weights_q.reshape((n_gals, 1, n_age))
    _w_lgmet_q = lgmet_weights_q.reshape((n_gals, n_met, 1))
    ssp_weights_q = _w_lgmet_q * _w_age_q

    _w_age_ms = smooth_age_weights_ms.reshape((n_gals, 1, n_age))
    _w_lgmet_ms = lgmet_weights_ms.reshape((n_gals, n_met, 1))
    ssp_weights_smooth_ms = _w_lgmet_ms * _w_age_ms

    _w_age_bursty_ms = bursty_age_weights_ms.reshape((n_gals, 1, n_age))
    ssp_weights_bursty_ms = _w_lgmet_ms * _w_age_bursty_ms

    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        z_obs, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    wave_eff_galpop = lc_phot_kern.interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    # Delta mags
    frac_ssp_err_q = lc_phot_kern.get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        diffstar_galpop.logsm_obs_q,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )
    frac_ssp_err_ms = lc_phot_kern.get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        diffstar_galpop.logsm_obs_ms,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )

    ran_key, dust_key = jran.split(ran_key, 2)
    av_key, delta_key, funo_key = jran.split(dust_key, 3)
    uran_av = jran.uniform(av_key, shape=(n_gals,))
    uran_delta = jran.uniform(delta_key, shape=(n_gals,))
    uran_funo = jran.uniform(funo_key, shape=(n_gals,))

    ftrans_args_q = (
        spspop_params.dustpop_params,
        wave_eff_galpop,
        diffstar_galpop.logsm_obs_q,
        diffstar_galpop.logssfr_obs_q,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = lc_phot_kern.calc_dust_ftrans_vmap(*ftrans_args_q)
    ftrans_q = _res[1]

    ftrans_args_ms = (
        spspop_params.dustpop_params,
        wave_eff_galpop,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = lc_phot_kern.calc_dust_ftrans_vmap(*ftrans_args_ms)
    ftrans_ms = _res[1]

    _mstar_ms = 10 ** diffstar_galpop.logsm_obs_ms.reshape((n_gals, 1))
    _mstar_q = 10 ** diffstar_galpop.logsm_obs_q.reshape((n_gals, 1))

    _w_smooth_ms = ssp_weights_smooth_ms.reshape((n_gals, 1, n_met, n_age))
    _w_bursty_ms = ssp_weights_bursty_ms.reshape((n_gals, 1, n_met, n_age))
    _w_q = ssp_weights_q.reshape((n_gals, 1, n_met, n_age))

    _ferr_ssp_ms = frac_ssp_err_ms.reshape((n_gals, n_bands, 1, 1))
    _ferr_ssp_q = frac_ssp_err_q.reshape((n_gals, n_bands, 1, 1))

    ran_key, ssp_q_key, ssp_ms_key = jran.split(ran_key, 3)
    delta_scatter_q = ssp_err_model.compute_delta_scatter(ssp_q_key, frac_ssp_err_q)
    delta_scatter_ms = ssp_err_model.compute_delta_scatter(ssp_ms_key, frac_ssp_err_ms)

    _ftrans_ms = ftrans_ms.reshape((n_gals, n_bands, 1, n_age))
    _ftrans_q = ftrans_q.reshape((n_gals, n_bands, 1, n_age))

    integrand_q = ssp_photflux_table * _w_q * _ftrans_q * _ferr_ssp_q
    photflux_galpop_q = jnp.sum(integrand_q, axis=(2, 3)) * _mstar_q
    obs_mags_q = -2.5 * jnp.log10(photflux_galpop_q) + delta_scatter_q

    integrand_smooth_ms = ssp_photflux_table * _w_smooth_ms * _ftrans_ms * _ferr_ssp_ms
    photflux_galpop_smooth_ms = jnp.sum(integrand_smooth_ms, axis=(2, 3)) * _mstar_ms
    obs_mags_smooth_ms = -2.5 * jnp.log10(photflux_galpop_smooth_ms) + delta_scatter_ms

    integrand_bursty_ms = ssp_photflux_table * _w_bursty_ms * _ftrans_ms * _ferr_ssp_ms
    photflux_galpop_bursty_ms = jnp.sum(integrand_bursty_ms, axis=(2, 3)) * _mstar_ms
    obs_mags_bursty_ms = -2.5 * jnp.log10(photflux_galpop_bursty_ms) + delta_scatter_ms

    weights_q = diffstar_galpop.frac_q
    weights_smooth_ms = (1 - diffstar_galpop.frac_q) * (1 - p_burst_ms)
    weights_bursty_ms = (1 - diffstar_galpop.frac_q) * p_burst_ms

    lc_phot = lc_phot_kern.LCPHOT_EMPTY._replace(
        obs_mags_bursty_ms=obs_mags_bursty_ms,
        obs_mags_smooth_ms=obs_mags_smooth_ms,
        obs_mags_q=obs_mags_q,
        weights_bursty_ms=weights_bursty_ms,
        weights_smooth_ms=weights_smooth_ms,
        weights_q=weights_q,
    )

    ran_key, smooth_sfh_key = jran.split(ran_key, 2)
    uran_smooth_sfh = jran.uniform(smooth_sfh_key, shape=(n_gals,))
    cuml_q = lc_phot.weights_q
    cuml_ms = lc_phot.weights_q + lc_phot.weights_smooth_ms
    mc_q = uran_smooth_sfh < cuml_q
    diffstar_params = mc_select_diffstar_params(
        diffstar_galpop.diffstar_params_q, diffstar_galpop.diffstar_params_ms, mc_q
    )

    mc_smooth_ms = (uran_smooth_sfh >= cuml_q) & (uran_smooth_sfh < cuml_ms)
    mc_bursty_ms = uran_smooth_sfh >= cuml_ms
    mc_smooth_ms = mc_smooth_ms.reshape((n_gals, 1, 1))
    mc_bursty_ms = mc_bursty_ms.reshape((n_gals, 1, 1))
    ssp_weights = jnp.copy(ssp_weights_q)
    ssp_weights = jnp.where(mc_smooth_ms, ssp_weights_smooth_ms, ssp_weights)
    ssp_weights = jnp.where(mc_bursty_ms, ssp_weights_bursty_ms, ssp_weights)

    delta_mag_sed_ms = ssp_err_interp(
        ssp_data.ssp_wave, delta_scatter_ms, wave_eff_galpop
    )
    delta_mag_sed_q = ssp_err_interp(
        ssp_data.ssp_wave, delta_scatter_q, wave_eff_galpop
    )
    frac_ssp_err_sed_ms = 10 ** (-0.4 * delta_mag_sed_ms)
    frac_ssp_err_sed_q = 10 ** (-0.4 * delta_mag_sed_q)

    n_wave = ssp_data.ssp_wave.size
    ssp_wave_galpop = jnp.tile(ssp_data.ssp_wave, n_gals).reshape((n_gals, n_wave))

    ftrans_sed_args_ms = (
        spspop_params.dustpop_params,
        ssp_wave_galpop,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = lc_phot_kern.calc_dust_ftrans_vmap(*ftrans_sed_args_ms)
    ftrans_sed_ms = _res[1]

    ftrans_sed_args_q = (
        spspop_params.dustpop_params,
        ssp_wave_galpop,
        diffstar_galpop.logsm_obs_q,
        diffstar_galpop.logssfr_obs_q,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = lc_phot_kern.calc_dust_ftrans_vmap(*ftrans_sed_args_q)
    ftrans_sed_q = _res[1]

    sed_info = SEDINFO_EMPTY._replace(
        diffstar_params=diffstar_params,
        burst_params=burst_params,
        ssp_weights=ssp_weights,
        wave_eff_galpop=wave_eff_galpop,
        frac_ssp_err_sed_ms=frac_ssp_err_sed_ms,
        frac_ssp_err_sed_q=frac_ssp_err_sed_q,
        ftrans_sed_ms=ftrans_sed_ms,
        ftrans_sed_q=ftrans_sed_q,
    )

    return lc_phot, sed_info
