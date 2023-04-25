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

from dsps.dust.utils import get_filter_effective_wavelength
from dsps.dust.att_curves import UV_BUMP_W0, UV_BUMP_DW
from dsps.dust.att_curves import _frac_transmission_from_k_lambda, sbl18_k_lambda

from .nagaraj22_dust import _get_median_dust_params_kern

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
    ssp_obsmag_table,
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
    weights, lgmet_weights, smooth_age_weights = _res

    ran_key, burst_key, att_curve_key = jran.split(ran_key, 3)

    gal_fburst, gal_dburst = _mc_burst(
        burst_key, gal_logsm_t_obs, gal_logssfr_t_obs, burst_params_pop
    )

    frac_trans = _compute_dust_transmission_fractions(
        att_curve_key,
        z_obs,
        gal_logsm_t_obs,
        gal_logssfr_t_obs,
        filter_waves,
        filter_trans,
        att_curve_params_pop,
        fracuno_pop_u_params,
    )

    ssp_lg_age_yr = ssp_lg_age + 9.0
    bursty_age_weights = _compute_bursty_age_weights_pop(
        ssp_lg_age_yr, smooth_age_weights, gal_fburst, gal_dburst
    )

    return weights, lgmet_weights, smooth_age_weights, bursty_age_weights, frac_trans


@jjit
def _mc_burst(ran_key, gal_logsm, gal_logssfr, params):
    n = gal_logsm.shape[0]
    fburst_key, dburst_key = jran.split(ran_key, 2)
    fburst = jran.uniform(fburst_key, minval=0, maxval=0.1, shape=(n,))
    dburst = jran.uniform(
        dburst_key, minval=DEFAULT_DBURST, maxval=DEFAULT_DBURST + 0.1, shape=(n,)
    )
    return fburst, dburst


@jjit
def mc_generate_dust_params_kern(
    ran_key, logsm, logssfr, redshift, att_curve_params_pop
):
    delta_key, av_key = jran.split(ran_key, 2)
    n = logsm.size

    tau_params_pop, delta_params_pop = att_curve_params_pop
    median_eb, median_delta, median_av = _get_median_dust_params_kern(
        logsm, logssfr, redshift, tau_params_pop, delta_params_pop
    )
    delta_lgav = jran.uniform(av_key, minval=-0.2, maxval=0.2, shape=(n,))
    lgav = delta_lgav + jnp.log10(median_av)
    gal_av = 10**lgav

    gal_delta = median_delta + jran.uniform(
        delta_key, minval=-0.1, maxval=0.1, shape=(n,)
    )
    gal_eb = median_eb + jran.uniform(delta_key, minval=-0.15, maxval=0.15, shape=(n,))

    gal_att_curve_params = jnp.array((gal_eb, gal_delta, gal_av)).T

    return gal_att_curve_params


@jjit
def _compute_dust_transmission_fractions(
    att_curve_key,
    z_obs,
    logsm_t_obs,
    logssfr_t_obs,
    filter_waves,
    filter_trans,
    att_curve_params_pop,
    fracuno_pop_u_params,
):
    gal_att_curve_params = mc_generate_dust_params_kern(
        att_curve_key, logsm_t_obs, logssfr_t_obs, z_obs, att_curve_params_pop
    )
    n_gals = logsm_t_obs.shape[0]
    gal_frac_unobscured = jnp.zeros(n_gals) + 0.01

    transmission_fractions = _get_effective_attenuation_vmap(
        filter_waves, filter_trans, z_obs, gal_att_curve_params, gal_frac_unobscured
    )
    return transmission_fractions.T


@jjit
def _get_effective_attenuation_sbl18(
    filter_wave, filter_trans, redshift, dust_params, frac_unobscured
):
    """Attenuation factor at the effective wavelength of the filter"""

    lambda_eff_angstrom = get_filter_effective_wavelength(
        filter_wave, filter_trans, redshift
    )
    lambda_eff_micron = lambda_eff_angstrom / 10_000

    dust_Eb, dust_delta, dust_Av = dust_params
    dust_x0_microns = UV_BUMP_W0
    bump_width_microns = UV_BUMP_DW
    k_lambda = sbl18_k_lambda(
        lambda_eff_micron, dust_x0_microns, bump_width_microns, dust_Eb, dust_delta
    )
    attenuation_factor = _frac_transmission_from_k_lambda(
        k_lambda, dust_Av, frac_unobscured
    )
    return attenuation_factor


_A = (None, None, None, 0, 0)
_B = (0, 0, None, None, None)
_get_effective_attenuation_vmap = jjit(
    vmap(vmap(_get_effective_attenuation_sbl18, _A), _B)
)
