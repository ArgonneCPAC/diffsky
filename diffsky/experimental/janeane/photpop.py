"""
"""
from dsps.sed import calc_ssp_weights_sfh_table_lognormal_mdf
from jax import vmap, jit as jjit
from jax import random as jran
from dsps.cosmology.flat_wcdm import _age_at_z_kern
from jax import numpy as jnp
from dsps.metallicity.mzr import mzr_model, DEFAULT_MZR_PDICT
from dsps.constants import SFR_MIN
from dsps.sed.stellar_age_weights import _calc_logsm_table_from_sfh_table
from dsps.experimental.diffburst import _compute_bursty_age_weights_pop
from dsps.experimental.diffburst import DEFAULT_DBURST

from .dustpop import _compute_dust_transmission_fractions
from .diffburstpop import _get_lgfburst

DEFAULT_MZR_PARAMS = jnp.array(list(DEFAULT_MZR_PDICT.values()))
_linterp_vmap = jjit(vmap(jnp.interp, in_axes=(None, None, 0)))

_g = (None, 0, 0, None, None, None, None)
calc_ssp_weights_sfh_table_lognormal_mdf_vmap = jjit(
    vmap(calc_ssp_weights_sfh_table_lognormal_mdf, in_axes=_g)
)

_b = (None, 0, None)
_calc_logsm_table_from_sfh_table_vmap = jjit(
    vmap(_calc_logsm_table_from_sfh_table, in_axes=_b)
)


@jjit
def get_obs_photometry_singlez(
    ran_key,
    filter_waves,
    filter_trans,
    ssp_obs_photflux_table,
    ssp_lgmet,
    ssp_lg_age,
    gal_t_table,
    gal_sfr_table,
    burst_params_pop,
    att_curve_params_pop,
    fracuno_pop_u_params,
    cosmo_params,
    z_obs,
    met_params=DEFAULT_MZR_PARAMS,
    lgmet_scatter=0.2,
):
    """Compute apparent magnitudes of galaxies at a single redshift

    Parameters
    ----------
    ran_key : jax.random.PRNGKey

    filter_waves : array of shape (n_filters, n_trans_curve)
        Wavelength of the filter transmission curves in Angstroms

    filter_trans : array of shape (n_filters, n_trans_curve)
        Transmission curves defining fractional transmission of the filters

    ssp_obs_photflux_table : ndarray of shape (n_met, n_age, n_filters)

    ssp_lgmet : ndarray of shape (n_met, )
        Array of log10(Z) of the SSP templates

    ssp_lg_age : ndarray of shape (n_ages, )
        Array of log10(age/Gyr) of the SSP templates

    gal_t_table : ndarray of shape (n_t, )
        Age of the universe in Gyr at which the input galaxy SFH and metallicity
        have been tabulated

    gal_sfr_table : ndarray of shape (n_gals, n_t)
        Star formation history in Msun/yr evaluated at the input gal_t_table

    burst_params_pop : ndarray of shape (n_burst_params, )

    att_curve_params_pop : ndarray of shape (n_att_curve_params, )

    fracuno_pop_u_params : ndarray of shape (n_funo_params, )

    cosmo_params : 4-element sequence, (Om0, w0, wa, h)

    z_obs : float

    Returns
    -------
    weights : ndarray of shape (n_gals, n_met, n_ages)
        SSP weights of the joint distribution of stellar age and metallicity

    lgmet_weights : ndarray of shape (n_gals, n_met)
        SSP weights of the distribution of stellar metallicity

    smooth_age_weights : ndarray of shape (n_gals, n_ages)
        SSP weights of the distribution of stellar age for the smooth SFH

    bursty_age_weights : ndarray of shape (n_gals, n_ages)
        SSP weights of the distribution of stellar age for the SFH that includes bursts

    frac_trans : ndarray of shape (n_gals, n_filters)
        Fraction of the flux transmitted through dust for each galaxy in each filter

    gal_obsflux_nodust : ndarray of shape (n_gals, n_filters)
        Flux in Lsun of each galaxy in each filter, ignoring dust attenuation
        gal_obsmags_nodust = -2.5*log10(gal_obsflux_nodust)

    gal_obsflux : ndarray of shape (n_gals, n_filters)
        Flux in Lsun of each galaxy in each filter, including dust attenuation
        gal_obsmags = -2.5*log10(gal_obsflux)

    """
    n_gals = gal_sfr_table.shape[0]
    n_met = ssp_lgmet.shape[0]
    n_age = ssp_lg_age.shape[0]
    n_filters = filter_waves.shape[0]

    t_obs = _age_at_z_kern(z_obs, *cosmo_params)
    lgt_obs = jnp.log10(t_obs)
    lgt_table = jnp.log10(gal_t_table)

    gal_sfr_table = jnp.where(gal_sfr_table < SFR_MIN, SFR_MIN, gal_sfr_table)
    gal_logsm_table = _calc_logsm_table_from_sfh_table_vmap(
        gal_t_table, gal_sfr_table, SFR_MIN
    )
    gal_logsfr_table = jnp.log10(gal_sfr_table)

    gal_logsm_t_obs = _linterp_vmap(lgt_obs, lgt_table, gal_logsm_table)
    gal_logsfr_t_obs = _linterp_vmap(lgt_obs, lgt_table, gal_logsfr_table)
    gal_logssfr_t_obs = gal_logsfr_t_obs - gal_logsm_t_obs

    gal_lgmet = mzr_model(gal_logsm_t_obs, t_obs, *met_params[:-1])

    _res = calc_ssp_weights_sfh_table_lognormal_mdf_vmap(
        gal_t_table,
        gal_sfr_table,
        gal_lgmet,
        lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age,
        t_obs,
    )
    lgmet_weights, smooth_age_weights = _res[1:]

    ran_key, burst_key, dust_key = jran.split(ran_key, 3)

    gal_lgf_burst = _get_lgfburst(gal_logsm_t_obs, gal_logssfr_t_obs, burst_params_pop)
    gal_fburst = 10**gal_lgf_burst
    gal_dburst = jnp.zeros(n_gals) + DEFAULT_DBURST

    ssp_lg_age_yr = ssp_lg_age + 9.0
    bursty_age_weights = _compute_bursty_age_weights_pop(
        ssp_lg_age_yr, smooth_age_weights, gal_fburst, gal_dburst
    )

    _w_age = bursty_age_weights.reshape((n_gals, 1, n_age))
    _w_met = lgmet_weights.reshape((n_gals, n_met, 1))
    _w = _w_age * _w_met
    _norm = jnp.sum(_w, axis=(1, 2))
    weights = _w / _norm

    _ssp_fluxes = ssp_obs_photflux_table.reshape((1, n_met, n_age, n_filters))
    w = weights.reshape((n_gals, n_met, n_age, 1))
    gal_obsflux_per_mstar = jnp.sum(w * _ssp_fluxes, axis=(1, 2))

    _gal_mstar = 10 ** gal_logsm_t_obs.reshape((n_gals, 1))
    gal_obsflux_nodust = gal_obsflux_per_mstar * _gal_mstar

    frac_trans = _compute_dust_transmission_fractions(
        dust_key,
        z_obs,
        gal_logsm_t_obs,
        gal_logssfr_t_obs,
        gal_lgf_burst,
        filter_waves,
        filter_trans,
        att_curve_params_pop,
        fracuno_pop_u_params,
    )
    gal_obsflux = gal_obsflux_nodust * frac_trans

    return (
        weights,
        lgmet_weights,
        smooth_age_weights,
        bursty_age_weights,
        frac_trans,
        gal_obsflux_nodust,
        gal_obsflux,
    )
