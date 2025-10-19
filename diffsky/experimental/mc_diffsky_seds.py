# flake8: noqa: E402
"""Module implements the mc_diffsky_seds Monte Carlo generator of lightcone SEDs"""
from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple

from diffmah import logmh_at_t_obs
from diffstar.diffstarpop.param_utils import mc_select_diffstar_params
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z0
from dsps.metallicity import umzr
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..burstpop import freqburst_mono
from ..param_utils import diffsky_param_wrapper as dpw
from ..ssp_err_model import ssp_err_model
from . import lc_phot_kern
from . import photometry_interpolation as photerp

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
SED_INFO_KEYS = ["rest_sed", *PHOT_INFO_KEYS, "frac_ssp_err_sed", "ftrans_sed"]

SedInfo = namedtuple("SedInfo", SED_INFO_KEYS)
SEDINFO_EMPTY = SedInfo._make([None] * len(SedInfo._fields))

PhotInfo = namedtuple("PhotInfo", PHOT_INFO_KEYS)
PHOTINFO_EMPTY = PhotInfo._make([None] * len(PhotInfo._fields))

ssp_err_interp = jjit(vmap(ssp_err_model._tw_wave_interp_kern, in_axes=(None, 0, 0)))


def mc_weighted_diffsky_lightcone(
    ran_key,
    lc_data,
    diffstarpop_params=dpw.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
    mzr_params=dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
    spspop_params=dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
    scatter_params=dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
    ssperr_params=dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
    cosmo_params=DEFAULT_COSMOLOGY,
):
    """Populate the input lightcone with galaxy SEDs

    Parameters
    ----------
    ran_key : jax.random.key

    lc_data : namedtuple
        Contains info about the halo lightcone, SED inputs, and diffsky parameters

    Returns
    -------
    sed_info : dict
        Contains info about the galaxy SEDs

    """
    param_collection = dpw.DEFAULT_PARAM_COLLECTION._make(
        (diffstarpop_params, mzr_params, spspop_params, scatter_params, ssperr_params)
    )
    sed_data = _mc_diffsky_seds_kern(
        ran_key, *lc_data[1:], *param_collection, cosmo_params
    )
    return sed_data


def _mc_diffsky_seds_flat_u_params(u_param_arr, ran_key, lc_data, cosmo_params):
    u_param_collection = dpw.get_u_param_collection_from_u_param_array(u_param_arr)
    param_collection = dpw.get_param_collection_from_u_param_collection(
        *u_param_collection
    )
    sed_data = _mc_diffsky_seds_kern(
        ran_key, *lc_data[1:], *param_collection, cosmo_params
    )
    return sed_data


def _mc_diffsky_phot_flat_u_params(u_param_arr, ran_key, lc_data, cosmo_params):
    u_param_collection = dpw.get_u_param_collection_from_u_param_array(u_param_arr)
    param_collection = dpw.get_param_collection_from_u_param_collection(
        *u_param_collection
    )
    phot_data = _mc_diffsky_phot_kern(
        ran_key, *lc_data[1:], *param_collection, cosmo_params
    )
    return phot_data


def _mc_diffsky_seds_kern(
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
    cosmo_params,
):
    """Populate the input lightcone with galaxy SEDs"""
    n_z_table, n_bands, n_met, n_age = precomputed_ssp_mag_table.shape
    n_gals = logmp0.size

    # Calculate halo mass at the observed redshift
    t0 = age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t0)
    logmp_obs = logmh_at_t_obs(mah_params, t_obs, lgt0)

    # Calculate SFH with diffstarpop
    ran_key, sfh_key = jran.split(ran_key, 2)
    diffstar_galpop = lc_phot_kern.diffstarpop_lc_cen_wrapper(
        diffstarpop_params, sfh_key, mah_params, logmp0, t_table, t_obs
    )
    # diffstar_galpop has separate diffstar params and SFH tables for ms and q

    # Calculate stellar age PDF weights from SFH
    smooth_age_weights_ms = lc_phot_kern.calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_ms, ssp_data.ssp_lg_age_gyr, t_obs
    )
    # smooth_age_weights_ms.shape = (n_gals, n_age)
    smooth_age_weights_q = lc_phot_kern.calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_q, ssp_data.ssp_lg_age_gyr, t_obs
    )

    # Calculate stellar age PDF weights from SFH + burstiness
    _args = (
        spspop_params.burstpop_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
        ssp_data.ssp_lg_age_gyr,
        smooth_age_weights_ms,
    )
    _res = lc_phot_kern._calc_bursty_age_weights_vmap(*_args)
    bursty_age_weights_ms = _res[0]  # bursty_age_weights_ms.shape = (n_gals, age)
    burst_params = _res[1]  # ('lgfburst', 'lgyr_peak', 'lgyr_max')

    # Calculate the frequency of SFH bursts
    p_burst_ms = freqburst_mono.get_freqburst_from_freqburst_params(
        spspop_params.burstpop_params.freqburst_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
    )

    # Calculate mean metallicity of the population
    lgmet_med_ms = umzr.mzr_model(diffstar_galpop.logsm_obs_ms, t_obs, *mzr_params)
    lgmet_med_q = umzr.mzr_model(diffstar_galpop.logsm_obs_q, t_obs, *mzr_params)

    # Calculate metallicity distribution function
    # lgmet_weights_q.shape = (n_gals, n_met)
    lgmet_weights_ms = lc_phot_kern._calc_lgmet_weights_galpop(
        lgmet_med_ms, lc_phot_kern.LGMET_SCATTER, ssp_data.ssp_lgmet
    )
    lgmet_weights_q = lc_phot_kern._calc_lgmet_weights_galpop(
        lgmet_med_q, lc_phot_kern.LGMET_SCATTER, ssp_data.ssp_lgmet
    )

    # Calculate SSP weights = P_SSP = P_met * P_age
    _w_age_ms = smooth_age_weights_ms.reshape((n_gals, 1, n_age))
    _w_lgmet_ms = lgmet_weights_ms.reshape((n_gals, n_met, 1))
    ssp_weights_smooth_ms = _w_lgmet_ms * _w_age_ms

    _w_age_bursty_ms = bursty_age_weights_ms.reshape((n_gals, 1, n_age))
    ssp_weights_bursty_ms = _w_lgmet_ms * _w_age_bursty_ms

    _w_age_q = smooth_age_weights_q.reshape((n_gals, 1, n_age))
    _w_lgmet_q = lgmet_weights_q.reshape((n_gals, n_met, 1))
    ssp_weights_q = _w_lgmet_q * _w_age_q  # (n_gals, n_met, n_age)

    # Interpolate SSP mag table to z_obs of each galaxy
    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        z_obs, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    # For each filter, calculate λ_eff in the restframe of each galaxy
    wave_eff_galpop = lc_phot_kern.interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_err_ms = lc_phot_kern.get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        diffstar_galpop.logsm_obs_ms,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )
    # frac_ssp_err_ms.shape = (n_gals, n_bands)
    frac_ssp_err_q = lc_phot_kern.get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        diffstar_galpop.logsm_obs_q,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )

    # Generate randoms for stochasticity in dust attenuation curves
    ran_key, dust_key = jran.split(ran_key, 2)
    av_key, delta_key, funo_key = jran.split(dust_key, 3)
    uran_av = jran.uniform(av_key, shape=(n_gals,))
    uran_delta = jran.uniform(delta_key, shape=(n_gals,))
    uran_funo = jran.uniform(funo_key, shape=(n_gals,))

    # Calculate fraction of flux transmitted through dust for each galaxy
    # Note that F_trans(λ_eff, τ_age) varies with stellar age τ_age
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
    ftrans_q = _res[1]  # ftrans_q.shape = (n_gals, n_bands, n_age)
    noisy_dust_params_q = _res[3]  # fields = ('av', 'delta', 'funo')

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
    noisy_dust_params_ms = _res[3]  # fields = ('av', 'delta', 'funo')

    # Calculate stochasticity in fractional changes to SSP fluxes
    ran_key, ssp_q_key, ssp_ms_key = jran.split(ran_key, 3)
    delta_scatter_q = ssp_err_model.compute_delta_scatter(ssp_q_key, frac_ssp_err_q)
    delta_scatter_ms = ssp_err_model.compute_delta_scatter(ssp_ms_key, frac_ssp_err_ms)

    # Calculate fractional changes to SSP fluxes
    frac_ssp_err_ms = frac_ssp_err_ms * 10 ** (-0.4 * delta_scatter_ms)
    frac_ssp_err_q = frac_ssp_err_q * 10 ** (-0.4 * delta_scatter_q)

    # Reshape arrays before calculating galaxy magnitudes
    _ferr_ssp_ms = frac_ssp_err_ms.reshape((n_gals, n_bands, 1, 1))
    _ferr_ssp_q = frac_ssp_err_q.reshape((n_gals, n_bands, 1, 1))

    _ftrans_ms = ftrans_ms.reshape((n_gals, n_bands, 1, n_age))
    _ftrans_q = ftrans_q.reshape((n_gals, n_bands, 1, n_age))

    _mstar_ms = 10 ** diffstar_galpop.logsm_obs_ms.reshape((n_gals, 1))
    _mstar_q = 10 ** diffstar_galpop.logsm_obs_q.reshape((n_gals, 1))

    _w_smooth_ms = ssp_weights_smooth_ms.reshape((n_gals, 1, n_met, n_age))
    _w_bursty_ms = ssp_weights_bursty_ms.reshape((n_gals, 1, n_met, n_age))
    _w_q = ssp_weights_q.reshape((n_gals, 1, n_met, n_age))

    # Calculate galaxy magnitudes as PDF-weighted sums
    integrand_smooth_ms = ssp_photflux_table * _w_smooth_ms * _ftrans_ms * _ferr_ssp_ms
    photflux_galpop_smooth_ms = jnp.sum(integrand_smooth_ms, axis=(2, 3)) * _mstar_ms
    obs_mags_smooth_ms = -2.5 * jnp.log10(photflux_galpop_smooth_ms)

    integrand_bursty_ms = ssp_photflux_table * _w_bursty_ms * _ftrans_ms * _ferr_ssp_ms
    photflux_galpop_bursty_ms = jnp.sum(integrand_bursty_ms, axis=(2, 3)) * _mstar_ms
    obs_mags_bursty_ms = -2.5 * jnp.log10(photflux_galpop_bursty_ms)

    integrand_q = ssp_photflux_table * _w_q * _ftrans_q * _ferr_ssp_q
    photflux_galpop_q = jnp.sum(integrand_q, axis=(2, 3)) * _mstar_q
    obs_mags_q = -2.5 * jnp.log10(photflux_galpop_q)

    weights_smooth_ms = (1 - diffstar_galpop.frac_q) * (1 - p_burst_ms)
    weights_bursty_ms = (1 - diffstar_galpop.frac_q) * p_burst_ms
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
    ssp_weights = jnp.copy(ssp_weights_q)
    ssp_weights = jnp.where(mc_smooth_ms, ssp_weights_smooth_ms, ssp_weights)
    ssp_weights = jnp.where(mc_bursty_ms, ssp_weights_bursty_ms, ssp_weights)
    # ssp_weights.shape = (n_gals, n_met, n_age)

    n_wave = ssp_data.ssp_wave.size
    ftrans_sed = _get_ftrans_sed(
        z_obs,
        mc_sfh_type,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
        diffstar_galpop.logsm_obs_q,
        diffstar_galpop.logssfr_obs_q,
        uran_av,
        uran_delta,
        uran_funo,
        ssp_data,
        spspop_params,
        scatter_params,
    )

    frac_ssp_err_sed = _get_frac_ssp_err_sed(
        ssp_data,
        z_obs,
        mc_sfh_type,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logsm_obs_q,
        wave_eff_galpop,
        ssp_err_pop_params,
    )

    # Reshape stellar mass used to normalize SED
    _mstar_ms = 10 ** diffstar_galpop.logsm_obs_ms.reshape((n_gals, 1))
    _mstar_q = 10 ** diffstar_galpop.logsm_obs_q.reshape((n_gals, 1))
    logsm_obs = jnp.where(
        mc_sfh_type > 0, diffstar_galpop.logsm_obs_ms, diffstar_galpop.logsm_obs_q
    )
    mstar_obs = 10 ** logsm_obs.reshape((n_gals, 1))

    # Calculate specific SFR at z_obs
    logssfr_obs = jnp.where(
        mc_sfh_type > 0, diffstar_galpop.logssfr_obs_ms, diffstar_galpop.logssfr_obs_q
    )

    # Reshape arrays storing weights and fluxes form SED integrand
    frac_ssp_err = frac_ssp_err_sed.reshape((n_gals, 1, 1, n_wave))
    flux_table = ssp_data.ssp_flux.reshape((1, n_met, n_age, n_wave))
    weights = ssp_weights.reshape((n_gals, n_met, n_age, 1))

    # Compute restframe SED as PDF-weighted sum of SSPs
    sed_integrand = flux_table * weights * ftrans_sed * frac_ssp_err
    rest_sed = jnp.sum(sed_integrand, axis=(1, 2)) * mstar_obs

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
        noisy_dust_params_q.av[:, 0, :],
        noisy_dust_params_ms.av[:, 0, :],
    )
    delta = jnp.where(
        msk_q, noisy_dust_params_q.delta[:, 0], noisy_dust_params_ms.delta[:, 0]
    )
    funo = jnp.where(
        msk_q, noisy_dust_params_q.funo[:, 0], noisy_dust_params_ms.funo[:, 0]
    )
    dust_params = noisy_dust_params_q._make((av, delta, funo))

    sed_info = SEDINFO_EMPTY._replace(
        rest_sed=rest_sed,
        logmp_obs=logmp_obs,
        logsm_obs=logsm_obs,
        logssfr_obs=logssfr_obs,
        sfh_table=sfh_table,
        obs_mags=obs_mags,
        diffstar_params=diffstar_params,
        mc_sfh_type=mc_sfh_type,
        burst_params=burst_params,
        dust_params=dust_params,
        ssp_weights=ssp_weights,
        uran_av=uran_av,
        uran_delta=uran_delta,
        uran_funo=uran_funo,
        logsm_obs_ms=diffstar_galpop.logsm_obs_ms,
        logsm_obs_q=diffstar_galpop.logsm_obs_q,
        logssfr_obs_ms=diffstar_galpop.logssfr_obs_ms,
        logssfr_obs_q=diffstar_galpop.logssfr_obs_q,
        delta_scatter_ms=delta_scatter_ms,
        delta_scatter_q=delta_scatter_q,
        frac_ssp_err_sed=frac_ssp_err_sed,
        ftrans_sed=ftrans_sed,
    )

    return sed_info._asdict()


def _mc_diffsky_phot_kern(
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
    cosmo_params,
):
    """Populate the input lightcone with galaxy SEDs"""
    n_z_table, n_bands, n_met, n_age = precomputed_ssp_mag_table.shape
    n_gals = logmp0.size

    # Calculate halo mass at the observed redshift
    t0 = age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t0)
    logmp_obs = logmh_at_t_obs(mah_params, t_obs, lgt0)

    # Calculate SFH with diffstarpop
    ran_key, sfh_key = jran.split(ran_key, 2)
    diffstar_galpop = lc_phot_kern.diffstarpop_lc_cen_wrapper(
        diffstarpop_params, sfh_key, mah_params, logmp0, t_table, t_obs
    )
    # diffstar_galpop has separate diffstar params and SFH tables for ms and q

    # Calculate stellar age PDF weights from SFH
    smooth_age_weights_ms = lc_phot_kern.calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_ms, ssp_data.ssp_lg_age_gyr, t_obs
    )
    # smooth_age_weights_ms.shape = (n_gals, n_age)
    smooth_age_weights_q = lc_phot_kern.calc_age_weights_from_sfh_table_vmap(
        t_table, diffstar_galpop.sfh_q, ssp_data.ssp_lg_age_gyr, t_obs
    )

    # Calculate stellar age PDF weights from SFH + burstiness
    _args = (
        spspop_params.burstpop_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
        ssp_data.ssp_lg_age_gyr,
        smooth_age_weights_ms,
    )
    _res = lc_phot_kern._calc_bursty_age_weights_vmap(*_args)
    bursty_age_weights_ms = _res[0]  # bursty_age_weights_ms.shape = (n_gals, age)
    burst_params = _res[1]  # ('lgfburst', 'lgyr_peak', 'lgyr_max')

    # Calculate the frequency of SFH bursts
    p_burst_ms = freqburst_mono.get_freqburst_from_freqburst_params(
        spspop_params.burstpop_params.freqburst_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
    )

    # Calculate mean metallicity of the population
    lgmet_med_ms = umzr.mzr_model(diffstar_galpop.logsm_obs_ms, t_obs, *mzr_params)
    lgmet_med_q = umzr.mzr_model(diffstar_galpop.logsm_obs_q, t_obs, *mzr_params)

    # Calculate metallicity distribution function
    # lgmet_weights_q.shape = (n_gals, n_met)
    lgmet_weights_ms = lc_phot_kern._calc_lgmet_weights_galpop(
        lgmet_med_ms, lc_phot_kern.LGMET_SCATTER, ssp_data.ssp_lgmet
    )
    lgmet_weights_q = lc_phot_kern._calc_lgmet_weights_galpop(
        lgmet_med_q, lc_phot_kern.LGMET_SCATTER, ssp_data.ssp_lgmet
    )

    # Calculate SSP weights = P_SSP = P_met * P_age
    _w_age_ms = smooth_age_weights_ms.reshape((n_gals, 1, n_age))
    _w_lgmet_ms = lgmet_weights_ms.reshape((n_gals, n_met, 1))
    ssp_weights_smooth_ms = _w_lgmet_ms * _w_age_ms

    _w_age_bursty_ms = bursty_age_weights_ms.reshape((n_gals, 1, n_age))
    ssp_weights_bursty_ms = _w_lgmet_ms * _w_age_bursty_ms

    _w_age_q = smooth_age_weights_q.reshape((n_gals, 1, n_age))
    _w_lgmet_q = lgmet_weights_q.reshape((n_gals, n_met, 1))
    ssp_weights_q = _w_lgmet_q * _w_age_q  # (n_gals, n_met, n_age)

    # Interpolate SSP mag table to z_obs of each galaxy
    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        z_obs, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    # For each filter, calculate λ_eff in the restframe of each galaxy
    wave_eff_galpop = lc_phot_kern.interp_vmap2(z_obs, z_phot_table, wave_eff_table)

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_err_ms = lc_phot_kern.get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        diffstar_galpop.logsm_obs_ms,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )
    # frac_ssp_err_ms.shape = (n_gals, n_bands)
    frac_ssp_err_q = lc_phot_kern.get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        diffstar_galpop.logsm_obs_q,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )

    # Generate randoms for stochasticity in dust attenuation curves
    ran_key, dust_key = jran.split(ran_key, 2)
    av_key, delta_key, funo_key = jran.split(dust_key, 3)
    uran_av = jran.uniform(av_key, shape=(n_gals,))
    uran_delta = jran.uniform(delta_key, shape=(n_gals,))
    uran_funo = jran.uniform(funo_key, shape=(n_gals,))

    # Calculate fraction of flux transmitted through dust for each galaxy
    # Note that F_trans(λ_eff, τ_age) varies with stellar age τ_age
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
    ftrans_q = _res[1]  # ftrans_q.shape = (n_gals, n_bands, n_age)
    noisy_dust_params_q = _res[3]  # fields = ('av', 'delta', 'funo')

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
    noisy_dust_params_ms = _res[3]  # fields = ('av', 'delta', 'funo')

    # Calculate stochasticity in fractional changes to SSP fluxes
    ran_key, ssp_q_key, ssp_ms_key = jran.split(ran_key, 3)
    delta_scatter_q = ssp_err_model.compute_delta_scatter(ssp_q_key, frac_ssp_err_q)
    delta_scatter_ms = ssp_err_model.compute_delta_scatter(ssp_ms_key, frac_ssp_err_ms)

    # Calculate fractional changes to SSP fluxes
    frac_ssp_err_ms = frac_ssp_err_ms * 10 ** (-0.4 * delta_scatter_ms)
    frac_ssp_err_q = frac_ssp_err_q * 10 ** (-0.4 * delta_scatter_q)

    # Reshape arrays before calculating galaxy magnitudes
    _ferr_ssp_ms = frac_ssp_err_ms.reshape((n_gals, n_bands, 1, 1))
    _ferr_ssp_q = frac_ssp_err_q.reshape((n_gals, n_bands, 1, 1))

    _ftrans_ms = ftrans_ms.reshape((n_gals, n_bands, 1, n_age))
    _ftrans_q = ftrans_q.reshape((n_gals, n_bands, 1, n_age))

    _mstar_ms = 10 ** diffstar_galpop.logsm_obs_ms.reshape((n_gals, 1))
    _mstar_q = 10 ** diffstar_galpop.logsm_obs_q.reshape((n_gals, 1))

    _w_smooth_ms = ssp_weights_smooth_ms.reshape((n_gals, 1, n_met, n_age))
    _w_bursty_ms = ssp_weights_bursty_ms.reshape((n_gals, 1, n_met, n_age))
    _w_q = ssp_weights_q.reshape((n_gals, 1, n_met, n_age))

    # Calculate galaxy magnitudes as PDF-weighted sums
    integrand_smooth_ms = ssp_photflux_table * _w_smooth_ms * _ftrans_ms * _ferr_ssp_ms
    photflux_galpop_smooth_ms = jnp.sum(integrand_smooth_ms, axis=(2, 3)) * _mstar_ms
    obs_mags_smooth_ms = -2.5 * jnp.log10(photflux_galpop_smooth_ms)

    integrand_bursty_ms = ssp_photflux_table * _w_bursty_ms * _ftrans_ms * _ferr_ssp_ms
    photflux_galpop_bursty_ms = jnp.sum(integrand_bursty_ms, axis=(2, 3)) * _mstar_ms
    obs_mags_bursty_ms = -2.5 * jnp.log10(photflux_galpop_bursty_ms)

    integrand_q = ssp_photflux_table * _w_q * _ftrans_q * _ferr_ssp_q
    photflux_galpop_q = jnp.sum(integrand_q, axis=(2, 3)) * _mstar_q
    obs_mags_q = -2.5 * jnp.log10(photflux_galpop_q)

    weights_smooth_ms = (1 - diffstar_galpop.frac_q) * (1 - p_burst_ms)
    weights_bursty_ms = (1 - diffstar_galpop.frac_q) * p_burst_ms
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
    ssp_weights = jnp.copy(ssp_weights_q)
    ssp_weights = jnp.where(mc_smooth_ms, ssp_weights_smooth_ms, ssp_weights)
    ssp_weights = jnp.where(mc_bursty_ms, ssp_weights_bursty_ms, ssp_weights)
    # ssp_weights.shape = (n_gals, n_met, n_age)

    # Reshape stellar mass used to normalize SED
    _mstar_ms = 10 ** diffstar_galpop.logsm_obs_ms.reshape((n_gals, 1))
    _mstar_q = 10 ** diffstar_galpop.logsm_obs_q.reshape((n_gals, 1))
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
        noisy_dust_params_q.av[:, 0, :],
        noisy_dust_params_ms.av[:, 0, :],
    )
    delta = jnp.where(
        msk_q, noisy_dust_params_q.delta[:, 0], noisy_dust_params_ms.delta[:, 0]
    )
    funo = jnp.where(
        msk_q, noisy_dust_params_q.funo[:, 0], noisy_dust_params_ms.funo[:, 0]
    )
    dust_params = noisy_dust_params_q._make((av, delta, funo))

    phot_info = PHOTINFO_EMPTY._replace(
        logmp_obs=logmp_obs,
        logsm_obs=logsm_obs,
        logssfr_obs=logssfr_obs,
        sfh_table=sfh_table,
        obs_mags=obs_mags,
        diffstar_params=diffstar_params,
        mc_sfh_type=mc_sfh_type,
        burst_params=burst_params,
        dust_params=dust_params,
        ssp_weights=ssp_weights,
        uran_av=uran_av,
        uran_delta=uran_delta,
        uran_funo=uran_funo,
        logsm_obs_ms=diffstar_galpop.logsm_obs_ms,
        logsm_obs_q=diffstar_galpop.logsm_obs_q,
        logssfr_obs_ms=diffstar_galpop.logssfr_obs_ms,
        logssfr_obs_q=diffstar_galpop.logssfr_obs_q,
        delta_scatter_ms=delta_scatter_ms,
        delta_scatter_q=delta_scatter_q,
    )

    return phot_info._asdict()


def _recompute_sed_from_phot_mock(
    redshift_true,
    ssp_data,
    logmp_obs,
    mc_sfh_type,
    ssp_weights,
    uran_av,
    uran_delta,
    uran_funo,
    logsm_obs_ms,
    logsm_obs_q,
    logssfr_obs_ms,
    logssfr_obs_q,
    ssp_err_pop_params,
    spspop_params,
    scatter_params,
    z_phot_table,
    wave_eff_table,
):
    # Reshape arrays storing weights and fluxes form SED integrand
    n_met, n_age, n_wave = ssp_data.ssp_flux.shape
    n_gals = logmp_obs.size

    # For each filter, calculate λ_eff in the restframe of each galaxy
    wave_eff_galpop = lc_phot_kern.interp_vmap2(
        redshift_true, z_phot_table, wave_eff_table
    )

    frac_ssp_err_sed = _get_frac_ssp_err_sed(
        ssp_data,
        redshift_true,
        mc_sfh_type,
        logsm_obs_ms,
        logsm_obs_q,
        wave_eff_galpop,
        ssp_err_pop_params,
    )

    ftrans_sed = _get_ftrans_sed(
        redshift_true,
        mc_sfh_type,
        logsm_obs_ms,
        logssfr_obs_ms,
        logsm_obs_q,
        logssfr_obs_q,
        uran_av,
        uran_delta,
        uran_funo,
        ssp_data,
        spspop_params,
        scatter_params,
    )

    frac_ssp_err = frac_ssp_err_sed.reshape((n_gals, 1, 1, n_wave))
    flux_table = ssp_data.ssp_flux.reshape((1, n_met, n_age, n_wave))
    weights = ssp_weights.reshape((n_gals, n_met, n_age, 1))

    # Compute restframe SED as PDF-weighted sum of SSPs
    logsm_obs = jnp.where(mc_sfh_type == 0, logsm_obs_q, logsm_obs_ms)
    mstar_obs = 10 ** logsm_obs.reshape((n_gals, 1))

    sed_integrand = flux_table * weights * ftrans_sed * frac_ssp_err
    rest_sed = jnp.sum(sed_integrand, axis=(1, 2)) * mstar_obs

    return rest_sed


def _recompute_photometry_from_phot_mock(
    redshift_true,
    ssp_data,
    logmp_obs,
    ssp_weights,
    uran_av,
    uran_delta,
    uran_funo,
    logsm_obs,
    logssfr_obs,
    ssp_err_pop_params,
    spspop_params,
    scatter_params,
    z_phot_table,
    wave_eff_table,
    precomputed_ssp_mag_table,
    delta_scatter,
):
    # Reshape arrays storing weights and fluxes form SED integrand
    n_z_table, n_bands, n_met, n_age = precomputed_ssp_mag_table.shape
    n_gals = logmp_obs.size

    # For each filter, calculate λ_eff in the restframe of each galaxy
    wave_eff_galpop = lc_phot_kern.interp_vmap2(
        redshift_true, z_phot_table, wave_eff_table
    )

    # Interpolate SSP mag table to z_obs of each galaxy
    photmag_table_galpop = photerp.interpolate_ssp_photmag_table(
        redshift_true, z_phot_table, precomputed_ssp_mag_table
    )
    ssp_photflux_table = 10 ** (-0.4 * photmag_table_galpop)

    frac_ssp_err = _get_frac_ssp_err_phot(
        redshift_true, logsm_obs, wave_eff_galpop, ssp_err_pop_params
    )
    # frac_ssp_err.shape = (n_gals, n_bands)

    # Calculate fractional changes to SSP fluxes
    frac_ssp_err = frac_ssp_err * 10 ** (-0.4 * delta_scatter)

    # Calculate fraction of flux transmitted through dust for each galaxy
    # Note that F_trans(λ_eff, τ_age) varies with stellar age τ_age
    ftrans_args = (
        spspop_params.dustpop_params,
        wave_eff_galpop,
        logsm_obs,
        logssfr_obs,
        redshift_true,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _ftrans_res = lc_phot_kern.calc_dust_ftrans_vmap(*ftrans_args)
    # ftrans, noisy_ftrans, dust_params, noisy_dust_params = _ftrans_res
    frac_trans = _ftrans_res[1]

    _ftrans = frac_trans.reshape((n_gals, n_bands, 1, n_age))

    # Reshape arrays before calculating galaxy magnitudes
    _ferr_ssp = frac_ssp_err.reshape((n_gals, n_bands, 1, 1))
    _w_ssp = ssp_weights.reshape((n_gals, 1, n_met, n_age))
    _mstar = 10 ** logsm_obs.reshape((n_gals, 1))

    # Calculate galaxy magnitudes as PDF-weighted sums
    integrand = ssp_photflux_table * _w_ssp * _ftrans * _ferr_ssp
    photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * _mstar
    obs_mags = -2.5 * jnp.log10(photflux_galpop)

    return obs_mags


def _get_ftrans_sed(
    z_obs,
    mc_sfh_type,
    logsm_obs_ms,
    logssfr_obs_ms,
    logsm_obs_q,
    logssfr_obs_q,
    uran_av,
    uran_delta,
    uran_funo,
    ssp_data,
    spspop_params,
    scatter_params,
):
    n_gals = z_obs.size
    n_wave = ssp_data.ssp_wave.size
    n_age = ssp_data.ssp_lg_age_gyr.size

    ssp_wave_galpop = jnp.tile(ssp_data.ssp_wave, n_gals).reshape((n_gals, n_wave))

    # Recalculate dust transmission fraction at each λ_SED for each galaxy
    ftrans_sed_args_ms = (
        spspop_params.dustpop_params,
        ssp_wave_galpop,
        logsm_obs_ms,
        logssfr_obs_ms,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = lc_phot_kern.calc_dust_ftrans_vmap(*ftrans_sed_args_ms)
    ftrans_sed_ms = _res[1]  # ftrans_sed_ms.shape = (n_gals, n_wave, n_age)

    ftrans_sed_args_q = (
        spspop_params.dustpop_params,
        ssp_wave_galpop,
        logsm_obs_q,
        logssfr_obs_q,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = lc_phot_kern.calc_dust_ftrans_vmap(*ftrans_sed_args_q)
    ftrans_sed_q = _res[1]  # ftrans_sed_q.shape = (n_gals, n_wave, n_age)

    # Select the transmission curve according to the SFH selection
    ftrans_sed = jnp.where(
        mc_sfh_type.reshape((n_gals, 1, 1)) > 0, ftrans_sed_ms, ftrans_sed_q
    )
    ftrans_sed = jnp.swapaxes(ftrans_sed, 1, 2)
    ftrans_sed = ftrans_sed.reshape((n_gals, 1, n_age, n_wave))

    return ftrans_sed


def _get_frac_ssp_err_sed(
    ssp_data,
    z_obs,
    mc_sfh_type,
    logsm_obs_ms,
    logsm_obs_q,
    wave_eff_galpop,
    ssp_err_pop_params,
):

    n_gals = mc_sfh_type.shape[0]

    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_err_ms = lc_phot_kern.get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        logsm_obs_ms,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )
    # frac_ssp_err_ms.shape = (n_gals, n_bands)
    frac_ssp_err_q = lc_phot_kern.get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        logsm_obs_q,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )

    # Calculate fractional change to restframe SED of each galaxy
    # F(λ_SED) is calculated by smooth triweight-interpolation of F(λ_eff)
    frac_ssp_err_sed_ms = ssp_err_interp(
        ssp_data.ssp_wave, frac_ssp_err_ms, wave_eff_galpop
    )
    # frac_ssp_err_sed_ms.shape = (n_gals, n_wave)
    frac_ssp_err_sed_q = ssp_err_interp(
        ssp_data.ssp_wave, frac_ssp_err_q, wave_eff_galpop
    )

    # Select the fractional change to SSP mags according to the SFH selection
    frac_ssp_err_sed = jnp.where(
        mc_sfh_type.reshape((n_gals, 1)) > 0, frac_ssp_err_sed_ms, frac_ssp_err_sed_q
    )

    return frac_ssp_err_sed


def _get_frac_ssp_err_phot(z_obs, logsm_obs, wave_eff_galpop, ssp_err_pop_params):
    # Calculate mean fractional change to the SSP fluxes in each band for each galaxy
    # L'_SSP(λ_eff) = L_SSP(λ_eff) & F_SSP(λ_eff)
    frac_ssp_err_phot = lc_phot_kern.get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        logsm_obs,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )
    return frac_ssp_err_phot  # frac_ssp_err_phot.shape = (n_gals, n_bands)
