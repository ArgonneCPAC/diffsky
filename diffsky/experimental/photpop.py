"""
"""
from dsps.sed import calc_ssp_weights_sfh_table_lognormal_mdf
from jax import vmap, jit as jjit
from dsps.cosmology.flat_wcdm import _age_at_z_kern
from jax import numpy as jnp
from dsps.metallicity.mzr import mzr_model, DEFAULT_MZR_PDICT
from dsps.constants import SFR_MIN
from dsps.sed.stellar_age_weights import _calc_logsm_table_from_sfh_table


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
    ssp_obsmag_table,
    ssp_lgmet,
    ssp_lg_age,
    gal_t_table,
    gal_sfr_table,
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

    smooth_ssp_weights = calc_ssp_weights_sfh_table_lognormal_mdf_vmap(
        gal_t_table,
        gal_sfr_table,
        gal_lgmet,
        lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age,
        t_obs,
    )
    return smooth_ssp_weights
