""""""

from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple

from diffstar.diffstarpop.param_utils import mc_select_diffstar_params
from dsps.metallicity import umzr
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ..burstpop import freqburst_mono
from ..ssp_err_model import ssp_err_model
from . import lc_phot_kern
from . import mc_diffstarpop_wrappers as mcdw
from . import photometry_interpolation as photerp
from .kernels.ssp_weight_kernels import compute_burstiness, get_smooth_ssp_weights
from .mc_diffstarpop_wrappers import N_T_TABLE

LGMET_SCATTER = 0.2


@jjit
def _mc_phot_kern(
    ran_key,
    z_obs,
    t_obs,
    mah_params,
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
    fb,
    n_t_table=N_T_TABLE,
):
    """Populate the input lightcone with galaxy SEDs"""
    n_z_table, n_bands, n_met, n_age = precomputed_ssp_mag_table.shape
    n_gals = z_obs.size

    # Calculate SFH with diffstarpop
    ran_key, sfh_key = jran.split(ran_key, 2)
    args = (diffstarpop_params, sfh_key, mah_params, t_obs, cosmo_params, fb)
    diffstar_galpop = mcdw.diffstarpop_cen_wrapper(*args, n_t_table=n_t_table)

    smooth_ssp_weights = get_smooth_ssp_weights(
        diffstar_galpop, ssp_data, t_obs, mzr_params, LGMET_SCATTER
    )

    burstiness = compute_burstiness(
        diffstar_galpop, smooth_ssp_weights, ssp_data, spspop_params.burstpop_params
    )

    # Calculate SSP weights = P_SSP = P_met * P_age
    _w_age_ms = smooth_ssp_weights.age_weights.ms.reshape((n_gals, 1, n_age))
    _w_lgmet_ms = smooth_ssp_weights.lgmet_weights.ms.reshape((n_gals, n_met, 1))
    ssp_weights_smooth_ms = _w_lgmet_ms * _w_age_ms

    _w_age_bursty_ms = burstiness.age_weights.reshape((n_gals, 1, n_age))
    ssp_weights_bursty_ms = _w_lgmet_ms * _w_age_bursty_ms

    _w_age_q = smooth_ssp_weights.age_weights.q.reshape((n_gals, 1, n_age))
    _w_lgmet_q = smooth_ssp_weights.lgmet_weights.q.reshape((n_gals, n_met, 1))
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

    weights_smooth_ms = (1 - diffstar_galpop.frac_q) * (1 - burstiness.p_burst)
    weights_bursty_ms = (1 - diffstar_galpop.frac_q) * burstiness.p_burst
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

    phot_info = PhotInfo(
        logmp_obs=diffstar_galpop.logmp_obs,
        logsm_obs=logsm_obs,
        logssfr_obs=logssfr_obs,
        sfh_table=sfh_table,
        obs_mags=obs_mags,
        diffstar_params=diffstar_params,
        mc_sfh_type=mc_sfh_type,
        burst_params=burstiness.burst_params,
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
PhotInfo = namedtuple("PhotInfo", PHOT_INFO_KEYS)
