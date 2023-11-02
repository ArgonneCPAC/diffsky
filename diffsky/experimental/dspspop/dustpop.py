"""
"""
from dsps.dust.att_curves import (
    UV_BUMP_DW,
    UV_BUMP_W0,
    _frac_transmission_from_k_lambda,
    sbl18_k_lambda,
)
from dsps.dust.utils import get_filter_effective_wavelength
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from .boris_dust import _get_funo_from_u_params_galpop
from .dust_deltapop import _get_dust_delta_galpop_from_u_params
from .lgavpop import _get_lgav_galpop_from_u_params


@jjit
def _median_dust_params_kern(
    ran_key, logsm, logssfr, lgav_u_params, dust_delta_u_params
):
    gal_lgav = _get_lgav_galpop_from_u_params(logsm, logssfr, lgav_u_params)
    gal_av = 10**gal_lgav
    gal_dust_delta = _get_dust_delta_galpop_from_u_params(
        logsm, logssfr, dust_delta_u_params
    )
    gal_eb = _eb_from_delta_kc13(gal_dust_delta)

    gal_att_curve_params = jnp.array((gal_eb, gal_dust_delta, gal_av)).T

    return gal_att_curve_params


@jjit
def _frac_dust_transmission_singlez_kernel(
    att_curve_key,
    z_obs,
    gal_logsm_t_obs,
    gal_logssfr_t_obs,
    gal_logfburst,
    ssp_lg_age_gyr,
    filter_waves,
    filter_trans,
    lgav_u_params,
    dust_delta_u_params,
    funo_u_params,
):
    """Calculate fraction of flux transmitted through dust for a collection of filters

    Parameters
    ----------
    att_curve_key : jax.random.PRNGKey

    z_obs : float

    gal_logsm_t_obs : ndarray, shape (n_gals, )

    gal_logssfr_t_obs : ndarray, shape (n_gals, )

    gal_logfburst : ndarray, shape (n_gals, )

    ssp_lg_age_gyr : ndarray, shape (n_age, )

    filter_waves : ndarray, shape (n_filters, n_wave)

    filter_trans : ndarray, shape (n_filters, n_wave)

    lgav_u_params : ndarray

    dust_delta_u_params : ndarray

    funo_u_params : ndarray

    Returns
    -------
    gal_frac_trans : ndarray, shape (n_gals, n_age, n_filters)

    """
    gal_att_curve_params = _median_dust_params_kern(
        att_curve_key,
        gal_logsm_t_obs,
        gal_logssfr_t_obs,
        lgav_u_params,
        dust_delta_u_params,
    )

    gal_frac_unobs = _get_funo_from_u_params_galpop(
        gal_logsm_t_obs, gal_logfburst, gal_logssfr_t_obs, ssp_lg_age_gyr, funo_u_params
    )

    gal_frac_trans = _get_effective_attenuation_vmap(
        filter_waves, filter_trans, z_obs, gal_att_curve_params, gal_frac_unobs
    )
    gal_frac_trans = jnp.swapaxes(gal_frac_trans, 0, 2)
    gal_frac_trans = jnp.swapaxes(gal_frac_trans, 0, 1)
    return gal_frac_trans, gal_att_curve_params, gal_frac_unobs


@jjit
def _frac_dust_transmission_lightcone_kernel(
    att_curve_key,
    gal_z_obs,
    gal_logsm_t_obs,
    gal_logssfr_t_obs,
    gal_logfburst,
    ssp_lg_age_gyr,
    filter_waves,
    filter_trans,
    lgav_u_params,
    dust_delta_u_params,
    funo_u_params,
):
    """Calculate fraction of flux transmitted through dust for galaxies on a lightcone
    observed through a collection of filters

    Parameters
    ----------
    att_curve_key : jax.random.PRNGKey

    gal_z_obs : ndarray, shape (n_gals, )

    gal_logsm_t_obs : ndarray, shape (n_gals, )

    gal_logssfr_t_obs : ndarray, shape (n_gals, )

    gal_logfburst : ndarray, shape (n_gals, )

    ssp_lg_age_gyr : ndarray, shape (n_age, )

    filter_waves : ndarray, shape (n_filters, n_wave)

    filter_trans : ndarray, shape (n_filters, n_wave)

    lgav_u_params : ndarray

    dust_delta_u_params : ndarray

    funo_u_params : ndarray

    Returns
    -------
    gal_frac_trans : ndarray, shape (n_gals, n_age, n_filters)

    gal_att_curve_params : ndarray, shape (n_gals, 3)

    gal_frac_unobs : ndarray, shape (n_gals, n_age)

    """
    gal_att_curve_params = _median_dust_params_kern(
        att_curve_key,
        gal_logsm_t_obs,
        gal_logssfr_t_obs,
        lgav_u_params,
        dust_delta_u_params,
    )

    gal_frac_unobs = _get_funo_from_u_params_galpop(
        gal_logsm_t_obs, gal_logfburst, gal_logssfr_t_obs, ssp_lg_age_gyr, funo_u_params
    )

    gal_frac_trans = _get_effective_attenuation_lightcone_vmap(
        filter_waves, filter_trans, gal_z_obs, gal_att_curve_params, gal_frac_unobs
    )
    gal_frac_trans = jnp.swapaxes(gal_frac_trans, 0, 2)
    gal_frac_trans = jnp.swapaxes(gal_frac_trans, 0, 1)
    return gal_frac_trans, gal_att_curve_params, gal_frac_unobs


@jjit
def _get_effective_attenuation_sbl18(
    filter_wave, filter_trans, redshift, att_curve_params, frac_unobscured
):
    """Attenuation factor at the effective wavelength of the filter"""

    lambda_eff_angstrom = get_filter_effective_wavelength(
        filter_wave, filter_trans, redshift
    )
    lambda_eff_micron = lambda_eff_angstrom / 10_000

    dust_Eb, dust_delta, dust_Av = att_curve_params
    k_lambda = sbl18_k_lambda(
        lambda_eff_micron,
        dust_Eb,
        dust_delta,
        UV_BUMP_W0,
        UV_BUMP_DW,
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

_C = (None, None, 0, 0, 0)
_get_effective_attenuation_lightcone_vmap = jjit(
    vmap(vmap(_get_effective_attenuation_sbl18, _C), _B)
)


@jjit
def _eb_from_delta_kc13(delta):
    return 0.85 - 1.9 * delta
