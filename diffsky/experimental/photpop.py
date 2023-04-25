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
    ssp_obsmag_table,
    ssp_lgmet,
    ssp_lg_age,
    gal_t_table,
    gal_sfr_table,
    burst_params,
    att_curve_params,
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
        burst_key, gal_logsm_t_obs, gal_logssfr_t_obs, burst_params
    )

    dust_params = mc_generate_dust_params_kern(
        att_curve_key,
        gal_logsm_t_obs,
        gal_logssfr_t_obs,
        z_obs,
        att_curve_params,
    )

    ssp_lg_age_yr = ssp_lg_age + 9.0
    bursty_age_weights = _compute_bursty_age_weights_pop(
        ssp_lg_age_yr, smooth_age_weights, gal_fburst, gal_dburst
    )

    return weights, lgmet_weights, smooth_age_weights, bursty_age_weights


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
def mc_generate_dust_params_kern(ran_key, logsm, logssfr, redshift, att_curve_params):
    delta_key, av_key = jran.split(ran_key, 2)
    n = logsm.size

    tau_params, delta_params = att_curve_params
    median_eb, median_delta, median_av = _get_median_dust_params_kern(
        logsm, logssfr, redshift, tau_params, delta_params
    )
    delta_lgav = jran.uniform(av_key, minval=-0.2, maxval=0.2, shape=(n,))
    lgav = delta_lgav + jnp.log10(median_av)
    av = 10**lgav

    delta = median_delta + jran.uniform(delta_key, minval=-0.1, maxval=0.1, shape=(n,))
    eb = median_eb + jran.uniform(delta_key, minval=-0.15, maxval=0.15, shape=(n,))

    dust_params = jnp.array((eb, delta, av)).T

    return dust_params
