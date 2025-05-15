""" """

from collections import namedtuple

from dsps.constants import SFR_MIN
from dsps.cosmology.flat_wcdm import _age_at_z_kern
from dsps.metallicity.mzr import DEFAULT_MET_PARAMS, mzr_model
from dsps.photometry.photometry_kernels import calc_obs_mag
from dsps.sed import calc_ssp_weights_sfh_table_lognormal_mdf
from dsps.utils import cumulative_mstar_formed
from jax import grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, vmap

from .burstpop import (
    DEFAULT_DIFFBURSTPOP_PARAMS,
    calc_bursty_age_weights_from_diffburstpop_params,
)
from .dustpop.tw_dustpop import (
    DEFAULT_DUSTPOP_PARAMS,
    calc_dust_ftrans_singlegal_multiwave_from_dustpop_params,
    calc_dust_ftrans_singlegal_singlewave_from_dustpop_params,
)
from .utils.tw_utils import _tw_gauss

LGMET_SCATTER = 0.20
TCURVE_WIDTH = 50.0

MICRON_PER_AA = 1 / 10_000

SPSPopParams = namedtuple("SPSPopParams", ["burstpop_params", "dustpop_params"])
DEFAULT_SPSPOP_PARAMS = SPSPopParams(
    DEFAULT_DIFFBURSTPOP_PARAMS, DEFAULT_DUSTPOP_PARAMS
)
NFILTER_WAVE = 1_000


@jjit
def calc_approx_singlemag_singlegal(
    spspop_params,
    cosmo_params,
    ssp_data,
    ssp_flux_table,
    wave_eff_aa,
    z_obs,
    t_table,
    sfr_table,
):
    """Calculate the photometry of a single galaxy through a single band

    Parameters
    ----------
    spspop_params : namedtuple

        burstpop_params : namedtuple

        dustpop_params : namedtuple

    cosmo_params : namedtuple
        (Om0, w0, wa, h)

    ssp_data : namedtuple

    ssp_flux_table : ndarray, shape (n_met, n_age)

    z_obs : float

    t_table : ndarray, shape (n_t, )

    sfr_table : ndarray, shape (n_t, )

    Returns
    -------
    mag : float
        Apparent magnitude of the galaxy

    """
    sfr_table = jnp.where(sfr_table < SFR_MIN, SFR_MIN, sfr_table)
    t_obs = _age_at_z_kern(z_obs, *cosmo_params)

    logsm_obs, logssfr_obs = _compute_tobs_properties(t_obs, t_table, sfr_table)
    lgmet = mzr_model(logsm_obs, t_obs, *DEFAULT_MET_PARAMS[:-1])

    ssp_weights = calc_ssp_weights_sfh_table_lognormal_mdf(
        t_table,
        sfr_table,
        lgmet,
        LGMET_SCATTER,
        ssp_data.ssp_lgmet,
        ssp_data.ssp_lg_age_gyr,
        t_obs,
    )
    smooth_age_weights = ssp_weights.age_weights

    age_weights, burst_params = calc_bursty_age_weights_from_diffburstpop_params(
        spspop_params.burstpop_params,
        logsm_obs,
        logssfr_obs,
        ssp_data.ssp_lg_age_gyr,
        smooth_age_weights,
    )

    frac_trans = calc_dust_ftrans_singlegal_singlewave_from_dustpop_params(
        spspop_params.dustpop_params,
        wave_eff_aa,
        logsm_obs,
        logssfr_obs,
        burst_params.lgfburst,
        ssp_data.ssp_lg_age_gyr,
    )

    gal_flux_table_nodust = ssp_flux_table * 10**logsm_obs
    gal_flux_table = gal_flux_table_nodust * frac_trans.reshape((1, *frac_trans.shape))

    w_age = age_weights.reshape((1, age_weights.size))
    w_age = w_age / w_age.sum()
    w_met = ssp_weights.lgmet_weights.reshape((ssp_weights.lgmet_weights.size, 1))
    w_met = w_met / w_age.sum()
    weights = w_met * w_age

    gal_flux = jnp.sum(gal_flux_table * weights, axis=(0, 1))
    gal_mag = -2.5 * jnp.log10(gal_flux)

    return gal_mag


@jjit
def _compute_tobs_properties(t_obs, t_table, sfr_table):
    lgt_obs = jnp.log10(t_obs)

    lgt_table = jnp.log10(t_table)
    mstar_table = cumulative_mstar_formed(t_table, sfr_table)
    logsm_table = jnp.log10(mstar_table)
    logsfr_table = jnp.log10(sfr_table)
    logsm_obs = jnp.interp(lgt_obs, lgt_table, logsm_table)
    logsfr_obs = jnp.interp(lgt_obs, lgt_table, logsfr_table)
    logssfr_obs = logsfr_obs - logsm_obs
    return logsm_obs, logssfr_obs


calc_approx_singlemag_singlegal_grads = jjit(
    value_and_grad(calc_approx_singlemag_singlegal, argnums=0)
)
_D = (None, None, None, 0, None, None, None, 0)
calc_approx_singlemag_galpop_grads = jjit(
    vmap(calc_approx_singlemag_singlegal_grads, in_axes=_D)
)


# Exact computations


@jjit
def get_fake_trans_curve(wave_eff_aa):
    wave_filter = jnp.linspace(wave_eff_aa - 200, wave_eff_aa + 200, NFILTER_WAVE)
    trans_filter = _tw_gauss(wave_filter, wave_eff_aa, TCURVE_WIDTH) * TCURVE_WIDTH * 2
    return wave_filter, trans_filter


@jjit
def calc_singlemag_singlegal(
    spspop_params,
    cosmo_params,
    ssp_data,
    wave_eff_aa,
    z_obs,
    t_table,
    sfr_table,
):
    """Calculate the photometry of a single galaxy through a single band

    Parameters
    ----------
    spspop_params : namedtuple

        burstpop_params : namedtuple

        dustpop_params : namedtuple

    cosmo_params : namedtuple
        (Om0, w0, wa, h)

    ssp_data : namedtuple

    wave_eff_aa : float

    z_obs : float

    t_table : ndarray, shape (n_t, )

    sfr_table : ndarray, shape (n_t, )

    Returns
    -------
    mag : float
        Apparent magnitude of the galaxy

    """
    sfr_table = jnp.where(sfr_table < SFR_MIN, SFR_MIN, sfr_table)
    t_obs = _age_at_z_kern(z_obs, *cosmo_params)

    logsm_obs, logssfr_obs = _compute_tobs_properties(t_obs, t_table, sfr_table)
    lgmet = mzr_model(logsm_obs, t_obs, *DEFAULT_MET_PARAMS[:-1])

    ssp_weights = calc_ssp_weights_sfh_table_lognormal_mdf(
        t_table,
        sfr_table,
        lgmet,
        LGMET_SCATTER,
        ssp_data.ssp_lgmet,
        ssp_data.ssp_lg_age_gyr,
        t_obs,
    )
    smooth_age_weights = ssp_weights.age_weights

    age_weights, burst_params = calc_bursty_age_weights_from_diffburstpop_params(
        spspop_params.burstpop_params,
        logsm_obs,
        logssfr_obs,
        ssp_data.ssp_lg_age_gyr,
        smooth_age_weights,
    )

    n_met = ssp_data.ssp_lgmet.shape[0]
    n_age = ssp_data.ssp_lg_age_gyr.shape[0]
    _x = ssp_weights.lgmet_weights.reshape((n_met, 1))
    _y = age_weights.reshape((1, n_age))
    weights = (_x * _y).reshape((n_met, n_age, 1))

    mstar_obs = 10**logsm_obs

    frac_trans = calc_dust_ftrans_singlegal_multiwave_from_dustpop_params(
        spspop_params.dustpop_params,
        ssp_data.ssp_wave,
        logsm_obs,
        logssfr_obs,
        burst_params.lgfburst,
        ssp_data.ssp_lg_age_gyr,
    )
    frac_trans = frac_trans.reshape((1, n_age, -1))

    gal_sed = jnp.sum(frac_trans * weights * ssp_data.ssp_flux, axis=(0, 1)) * mstar_obs

    wave_filter, trans_filter = get_fake_trans_curve(wave_eff_aa)

    obs_mag = calc_obs_mag(
        ssp_data.ssp_wave, gal_sed, wave_filter, trans_filter, z_obs, *cosmo_params
    )

    return obs_mag


calc_singlemag_singlegal_grads = jjit(grad(calc_singlemag_singlegal, argnums=0))
_D = (None, None, None, None, 0, None, 0)
calc_singlemag_galpop_grads = jjit(vmap(calc_singlemag_singlegal_grads, in_axes=_D))
_E = (None, None, None, 0, None, None, None)
calc_multimag_galpop_grads = jjit(vmap(calc_singlemag_galpop_grads, in_axes=_E))
