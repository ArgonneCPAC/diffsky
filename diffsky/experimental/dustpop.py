"""
"""
from jax import jit as jjit
from jax import random as jran
from jax import numpy as jnp
from jax import vmap
from dsps.dust.utils import get_filter_effective_wavelength
from dsps.dust.att_curves import UV_BUMP_W0, UV_BUMP_DW
from dsps.dust.att_curves import _frac_transmission_from_k_lambda, sbl18_k_lambda

from .nagaraj22_dust import _get_median_dust_params_kern
from .obscurepop_burst import mc_funobs


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
    dust_key,
    z_obs,
    logsm_t_obs,
    logssfr_t_obs,
    gal_lgfburst,
    filter_waves,
    filter_trans,
    att_curve_params_pop,
    lgfuno_pop_u_params,
):
    att_curve_key, funo_key = jran.split(dust_key, 2)
    gal_att_curve_params = mc_generate_dust_params_kern(
        att_curve_key, logsm_t_obs, logssfr_t_obs, z_obs, att_curve_params_pop
    )

    gal_frac_unobs = mc_funobs(funo_key, logsm_t_obs, gal_lgfburst, lgfuno_pop_u_params)

    gal_frac_trans = _get_effective_attenuation_vmap(
        filter_waves, filter_trans, z_obs, gal_att_curve_params, gal_frac_unobs
    )
    return gal_frac_trans.T


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
