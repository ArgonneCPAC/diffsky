"""
"""
from dsps.constants import SFR_MIN
from dsps.cosmology.flat_wcdm import _age_at_z_kern
from dsps.metallicity.mzr import DEFAULT_MET_PDICT, mzr_model
from dsps.sed import calc_ssp_weights_sfh_table_lognormal_mdf
from dsps.sed.stellar_age_weights import _calc_logsm_table_from_sfh_table
from dsps.sfh.diffburst import _pureburst_age_weights_from_u_params
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .burstshapepop import _get_burstshape_galpop_from_params
from .dustpop import _frac_dust_transmission_singlez_kernel
from .lgfburstpop import _get_lgfburst_galpop_from_u_params

_A = (None, 0, 0)
_pureburst_age_weights_from_u_params_vmap = jjit(
    vmap(_pureburst_age_weights_from_u_params, in_axes=_A)
)

DEFAULT_MET_PARAMS = jnp.array(list(DEFAULT_MET_PDICT.values()))
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
    ssp_lg_age_gyr,
    gal_t_table,
    gal_sfr_table,
    lgfburst_pop_u_params,
    burstshapepop_u_params,
    lgav_u_params,
    dust_delta_u_params,
    fracuno_pop_u_params,
    cosmo_params,
    z_obs,
    met_params=DEFAULT_MET_PARAMS,
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

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        Array of log10(age/Gyr) of the SSP templates

    gal_t_table : ndarray of shape (n_t, )
        Age of the universe in Gyr at which the input galaxy SFH and metallicity
        have been tabulated

    gal_sfr_table : ndarray of shape (n_gals, n_t)
        Star formation history in Msun/yr evaluated at the input gal_t_table

    lgfburst_pop_u_params : ndarray of shape (n_lgfburst_params, )
        Unbounded parameters of the lgfburstpop model

    burstshapepop_u_params : ndarray of shape (n_burstshapepop_params, )
        Unbounded parameters of the burstshapepop model

    lgav_u_params : ndarray of shape (n_lgavpop_params, )
        Unbounded parameters of the lgav model

    dust_delta_u_params : ndarray of shape (n_dust_deltapop_params, )
        Unbounded parameters of the dust_delta model

    fracuno_pop_u_params : ndarray of shape (n_funo_params, )
        unbounded parameters of the boris_dust model

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

    frac_trans : ndarray of shape (n_gals, n_ages, n_filters)
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
    n_age = ssp_lg_age_gyr.shape[0]
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

    mzr_params, lgmet_scatter = met_params[:-1], met_params[-1]
    gal_lgmet = mzr_model(gal_logsm_t_obs, t_obs, *mzr_params)

    _res = calc_ssp_weights_sfh_table_lognormal_mdf_vmap(
        gal_t_table,
        gal_sfr_table,
        gal_lgmet,
        lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age_gyr,
        t_obs,
    )
    lgmet_weights, smooth_age_weights = _res[1:]

    ran_key, burst_key, dust_key = jran.split(ran_key, 3)

    gal_lgf_burst = _get_lgfburst_galpop_from_u_params(
        gal_logsm_t_obs, gal_logssfr_t_obs, lgfburst_pop_u_params
    )
    gal_fburst = 10**gal_lgf_burst

    burstshape_u_params = _get_burstshape_galpop_from_params(
        gal_logsm_t_obs, gal_logssfr_t_obs, burstshapepop_u_params
    )
    burstshape_u_params = jnp.array(burstshape_u_params).T
    ssp_lg_age_yr = ssp_lg_age_gyr + 9
    burst_weights = _pureburst_age_weights_from_u_params_vmap(
        ssp_lg_age_yr, burstshape_u_params[:, 0], burstshape_u_params[:, 1]
    )

    _fb = gal_fburst.reshape((n_gals, 1))
    bursty_age_weights = _fb * burst_weights + (1 - _fb) * smooth_age_weights

    _w_age = bursty_age_weights.reshape((n_gals, 1, n_age))
    _w_met = lgmet_weights.reshape((n_gals, n_met, 1))
    _w = _w_age * _w_met
    _norm = jnp.sum(_w, axis=(1, 2))
    weights = _w / _norm.reshape((n_gals, 1, 1))

    frac_trans, att_curve_params, frac_unobs = _frac_dust_transmission_singlez_kernel(
        dust_key,
        z_obs,
        gal_logsm_t_obs,
        gal_logssfr_t_obs,
        gal_lgf_burst,
        ssp_lg_age_gyr,
        filter_waves,
        filter_trans,
        lgav_u_params,
        dust_delta_u_params,
        fracuno_pop_u_params,
    )

    _ssp_fluxes = ssp_obs_photflux_table.reshape((1, n_met, n_age, n_filters))
    w = weights.reshape((n_gals, n_met, n_age, 1))
    ft = frac_trans.reshape((n_gals, 1, n_age, n_filters))
    gal_obsflux_per_mstar_nodust = jnp.sum(w * _ssp_fluxes, axis=(1, 2))
    gal_obsflux_per_mstar = jnp.sum(w * _ssp_fluxes * ft, axis=(1, 2))

    _gal_mstar = 10 ** gal_logsm_t_obs.reshape((n_gals, 1))
    gal_obsflux_nodust = gal_obsflux_per_mstar_nodust * _gal_mstar
    gal_obsflux = gal_obsflux_per_mstar * _gal_mstar

    return (
        weights,
        lgmet_weights,
        smooth_age_weights,
        bursty_age_weights,
        frac_trans,
        gal_obsflux_nodust,
        gal_obsflux,
    )
