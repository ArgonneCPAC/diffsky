""""""

from collections import namedtuple
from functools import partial

from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstar.diffstarpop.param_utils import mc_select_diffstar_params
from dsps.metallicity import umzr
from dsps.sed import metallicity_weights as zmetw
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS, LGFBURST_MIN
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ...burstpop import diffqburstpop_mono, freqburst_mono
from ...dustpop import tw_dustpop_mono_noise
from ...dustpop.tw_dust import DEFAULT_DUST_PARAMS
from ...ssp_err_model import ssp_err_model

_M = (0, None, None)
_calc_lgmet_weights_galpop = jjit(
    vmap(zmetw.calc_lgmet_weights_from_lognormal_mdf, in_axes=_M)
)

_AGEPOP = (None, 0, None, 0)
calc_age_weights_from_sfh_table_vmap = jjit(
    vmap(calc_age_weights_from_sfh_table, in_axes=_AGEPOP)
)

_B = (None, 0, 0, None, 0)
_calc_bursty_age_weights_vmap = jjit(
    vmap(
        diffqburstpop_mono.calc_bursty_age_weights_from_diffburstpop_params, in_axes=_B
    )
)

_F = (None, None, None, 0, None)
_G = (None, 0, 0, 0, None)
get_frac_ssp_err_vmap = jjit(
    vmap(vmap(ssp_err_model.F_sps_err_lambda, in_axes=_F), in_axes=_G)
)

_D = (None, 0, None, None, None, None, None, None, None, None)
vmap_kern1 = jjit(
    vmap(
        tw_dustpop_mono_noise.calc_ftrans_singlegal_singlewave_from_dustpop_params,
        in_axes=_D,
    )
)
_E = (None, 0, 0, 0, 0, None, 0, 0, 0, None)
calc_dust_ftrans_vmap = jjit(vmap(vmap_kern1, in_axes=_E))

MSQ = namedtuple("MSQ", ("ms", "q"))
QMSB = namedtuple("QMSB", ("q", "smooth_ms", "bursty_ms"))

AgeWeights = namedtuple("AgeWeights", ("ms", "q"))
MetWeights = namedtuple("MetWeights", ("ms", "q"))
SSPWeights = namedtuple("SSPWeights", ("weights", "age_weights", "lgmet_weights"))
Burstiness = namedtuple(
    "Burstiness", ("age_weights", "weights", "burst_params", "p_burst")
)
FracSSPErr = namedtuple("FracSSPErr", ("ms", "q"))

DustAttenuation = namedtuple(
    "DustAttenuation", ("frac_trans", "dust_params", "dust_scatter")
)


@jjit
def get_age_weights_smooth(t_table, sfh_table, ssp_data, t_obs):
    age_weights = calc_age_weights_from_sfh_table_vmap(
        t_table, sfh_table, ssp_data.ssp_lg_age_gyr, t_obs
    )
    return age_weights


@jjit
def get_lgmet_weights(logsm_obs, ssp_data, t_obs, mzr_params, lgmet_scatter):
    # Calculate mean metallicity of the population
    lgmet_med = umzr.mzr_model(logsm_obs, t_obs, *mzr_params)

    # Compute metallicity distribution function
    lgmet_weights = _calc_lgmet_weights_galpop(
        lgmet_med, lgmet_scatter, ssp_data.ssp_lgmet
    )
    return lgmet_weights


@jjit
def get_smooth_ssp_weights(
    t_table, sfh_table, logsm_obs, ssp_data, t_obs, mzr_params, lgmet_scatter
):
    age_weights_smooth = get_age_weights_smooth(t_table, sfh_table, ssp_data, t_obs)
    lgmet_weights = get_lgmet_weights(
        logsm_obs, ssp_data, t_obs, mzr_params, lgmet_scatter
    )
    return age_weights_smooth, lgmet_weights


@jjit
def combine_age_met_weights(age_weights, lgmet_weights):
    n_gals, n_age = age_weights.shape
    n_met = lgmet_weights.shape[1]
    _w_age = age_weights.reshape((n_gals, 1, n_age))
    _w_lgmet = lgmet_weights.reshape((n_gals, n_met, 1))
    weights = _w_lgmet * _w_age
    return weights


@jjit
def compute_burstiness(
    uran_pburst,
    mc_is_q,
    logsm_obs,
    logssfr_obs,
    age_weights_smooth,
    lgmet_weights,
    ssp_data,
    burstpop_params,
):
    n_gals = mc_is_q.shape[0]

    _args = (
        burstpop_params,
        logsm_obs,
        logssfr_obs,
        ssp_data.ssp_lg_age_gyr,
        age_weights_smooth,
    )
    _res = _calc_bursty_age_weights_vmap(*_args)
    age_weights_bursty, burst_params = _res

    # Calculate the frequency of SFH bursts
    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        burstpop_params.freqburst_params, logsm_obs, logssfr_obs
    )

    mc_sfh_type = jnp.where(mc_is_q, 0, 1).astype(int)
    msk_bursty = (uran_pburst < p_burst) & (mc_sfh_type == 1)
    mc_sfh_type = jnp.where(msk_bursty, 2, mc_sfh_type).astype(int)

    lgfburst = jnp.where(mc_sfh_type < 2, LGFBURST_MIN + 0.01, burst_params.lgfburst)
    burst_params = burst_params._replace(lgfburst=lgfburst)

    age_weights = jnp.where(
        mc_sfh_type.reshape((n_gals, 1)) == 2, age_weights_bursty, age_weights_smooth
    )

    ssp_weights = combine_age_met_weights(age_weights, lgmet_weights)

    return ssp_weights, burst_params, mc_sfh_type


@jjit
def compute_frac_ssp_errors(ssp_err_pop_params, z_obs, logsm_obs, wave_eff_galpop):
    frac_ssp_err = get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        logsm_obs,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )

    return frac_ssp_err


@partial(jjit, static_argnames=["n_gals"])
def get_dust_randoms(dust_key, n_gals):

    # Generate randoms for stochasticity in dust attenuation curves
    av_key, delta_key, funo_key = jran.split(dust_key, 3)
    uran_av = jran.uniform(av_key, shape=(n_gals,))
    uran_delta = jran.uniform(delta_key, shape=(n_gals,))
    uran_funo = jran.uniform(funo_key, shape=(n_gals,))
    fields = "uran_av", "uran_delta", "uran_funo"
    DustRandoms = namedtuple("DustRandoms", fields)
    return DustRandoms(uran_av, uran_delta, uran_funo)


@partial(jjit, static_argnames=["n_gals"])
def get_burstiness_randoms(burst_key, n_gals):
    uran_pburst = jran.uniform(burst_key, shape=(n_gals,))
    return uran_pburst


@jjit
def compute_dust_attenuation(
    uran_av,
    uran_delta,
    uran_funo,
    logsm_obs,
    logssfr_obs,
    ssp_data,
    z_obs,
    wave_eff_galpop,
    dustpop_params,
    scatter_params,
):
    # Calculate fraction of flux transmitted through dust for each galaxy
    # Note that F_trans(λ_eff, τ_age) varies with stellar age τ_age
    ftrans_args = (
        dustpop_params,
        wave_eff_galpop,
        logsm_obs,
        logssfr_obs,
        z_obs,
        ssp_data.ssp_lg_age_gyr,
        uran_av,
        uran_delta,
        uran_funo,
        scatter_params,
    )
    _res = calc_dust_ftrans_vmap(*ftrans_args)
    frac_trans = _res[1]  # ftrans_q.shape = (n_gals, n_bands, n_age)
    dust_params = _res[3]  # fields = ('av', 'delta', 'funo')

    return frac_trans, dust_params


@jjit
def compute_obs_mags_ms_q(
    diffstar_galpop,
    dust_att,
    frac_ssp_errors,
    ssp_photflux_table,
    ssp_weights_smooth_ms,
    ssp_weights_bursty_ms,
    ssp_weights_q,
    delta_scatter_ms,
    delta_scatter_q,
):
    n_gals = diffstar_galpop.logsm_obs_ms.size
    n_gals, n_bands, n_met, n_age = ssp_photflux_table.shape

    # Calculate fractional changes to SSP fluxes
    frac_ssp_err_ms = frac_ssp_errors.ms * 10 ** (-0.4 * delta_scatter_ms)
    frac_ssp_err_q = frac_ssp_errors.q * 10 ** (-0.4 * delta_scatter_q)

    # Reshape arrays before calculating galaxy magnitudes
    _ferr_ssp_ms = frac_ssp_err_ms.reshape((n_gals, n_bands, 1, 1))
    _ferr_ssp_q = frac_ssp_err_q.reshape((n_gals, n_bands, 1, 1))

    _ftrans_ms = dust_att.frac_trans.ms.reshape((n_gals, n_bands, 1, n_age))
    _ftrans_q = dust_att.frac_trans.q.reshape((n_gals, n_bands, 1, n_age))

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

    return QMSB(
        q=obs_mags_q, smooth_ms=obs_mags_smooth_ms, bursty_ms=obs_mags_bursty_ms
    )


@jjit
def _compute_obs_mags_from_weights(
    logsm_obs,
    frac_trans,
    frac_ssp_errors,
    ssp_photflux_table,
    ssp_weights,
    delta_scatter,
):
    n_gals = logsm_obs.size
    n_gals, n_bands, n_met, n_age = ssp_photflux_table.shape

    # Calculate fractional changes to SSP fluxes
    frac_ssp_err = frac_ssp_errors * 10 ** (-0.4 * delta_scatter)

    # Reshape arrays before calculating galaxy magnitudes
    _ferr_ssp = frac_ssp_err.reshape((n_gals, n_bands, 1, 1))
    _ftrans = frac_trans.reshape((n_gals, n_bands, 1, n_age))
    _weights = ssp_weights.reshape((n_gals, 1, n_met, n_age))
    _mstar = 10 ** logsm_obs.reshape((n_gals, 1))

    # Calculate galaxy magnitudes as PDF-weighted sums
    integrand = ssp_photflux_table * _weights * _ftrans * _ferr_ssp
    photflux_galpop = jnp.sum(integrand, axis=(2, 3)) * _mstar
    obs_mags = -2.5 * jnp.log10(photflux_galpop)

    return obs_mags


@jjit
def compute_mc_realization(
    diffstar_galpop,
    burstiness,
    smooth_ssp_weights,
    dust_att,
    obs_mags,
    delta_scatter_ms,
    delta_scatter_q,
    ran_key,
):
    n_gals = diffstar_galpop.logmp_obs.size

    weights_smooth_ms = (1 - diffstar_galpop.frac_q) * (1 - burstiness.p_burst)
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
    ssp_weights = jnp.copy(smooth_ssp_weights.weights.q)
    ssp_weights = jnp.where(mc_smooth_ms, smooth_ssp_weights.weights.ms, ssp_weights)
    ssp_weights = jnp.where(mc_bursty_ms, burstiness.weights.ms, ssp_weights)
    # ssp_weights.shape = (n_gals, n_met, n_age)

    lgmet_weights = jnp.where(
        mc_sfh_type.reshape((n_gals, 1)) == 0,
        smooth_ssp_weights.lgmet_weights.q,
        smooth_ssp_weights.lgmet_weights.ms,
    )

    # Reshape stellar mass used to normalize SED
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
    mc_obs_mags = jnp.copy(obs_mags.q)
    mc_obs_mags = jnp.where(msk_ms, obs_mags.smooth_ms, mc_obs_mags)
    mc_obs_mags = jnp.where(msk_bursty, obs_mags.bursty_ms, mc_obs_mags)

    msk_q = mc_sfh_type == 0
    av = jnp.where(
        msk_q.reshape((n_gals, 1)),
        dust_att.dust_params.q.av[:, 0, :],
        dust_att.dust_params.ms.av[:, 0, :],
    )
    delta = jnp.where(
        msk_q, dust_att.dust_params.q.delta[:, 0], dust_att.dust_params.ms.delta[:, 0]
    )
    funo = jnp.where(
        msk_q, dust_att.dust_params.q.funo[:, 0], dust_att.dust_params.ms.funo[:, 0]
    )
    dust_params = dust_att.dust_params.q._make((av, delta, funo))

    lgfburst = jnp.where(
        msk_bursty,
        burstiness.burst_params.lgfburst,
        LGFBURST_MIN + 0.01,
    )
    lgyr_peak = jnp.where(
        msk_q,
        burstiness.burst_params.lgyr_peak,
        burstiness.burst_params.lgyr_peak,
    )
    lgyr_max = jnp.where(
        msk_q,
        burstiness.burst_params.lgyr_max,
        burstiness.burst_params.lgyr_max,
    )
    burst_params = burstiness.burst_params._replace(
        lgfburst=lgfburst, lgyr_peak=lgyr_peak, lgyr_max=lgyr_max
    )

    phot_info = MCPhotInfo(
        logmp_obs=diffstar_galpop.logmp_obs,
        logsm_obs=logsm_obs,
        logssfr_obs=logssfr_obs,
        sfh_table=sfh_table,
        obs_mags=mc_obs_mags,
        **diffstar_params._asdict(),
        mc_sfh_type=mc_sfh_type,
        burstiness=burstiness,
        **burst_params._asdict(),
        **dust_params._asdict(),
        ssp_weights=ssp_weights,
        lgmet_weights=lgmet_weights,
        uran_av=dust_att.dust_scatter.av,
        uran_delta=dust_att.dust_scatter.delta,
        uran_funo=dust_att.dust_scatter.funo,
        logsm_obs_ms=diffstar_galpop.logsm_obs_ms,
        logsm_obs_q=diffstar_galpop.logsm_obs_q,
        logssfr_obs_ms=diffstar_galpop.logssfr_obs_ms,
        logssfr_obs_q=diffstar_galpop.logssfr_obs_q,
        delta_scatter_ms=delta_scatter_ms,
        delta_scatter_q=delta_scatter_q,
        t_table=diffstar_galpop.t_table,
    )
    return phot_info


MCPHOT_INFO_KEYS = (
    "logmp_obs",
    "logsm_obs",
    "logssfr_obs",
    "sfh_table",
    "obs_mags",
    *DEFAULT_DIFFSTAR_PARAMS._fields,
    "mc_sfh_type",
    "burstiness",
    *DEFAULT_BURST_PARAMS._fields,
    *DEFAULT_DUST_PARAMS._fields,
    "ssp_weights",
    "lgmet_weights",
    "uran_av",
    "uran_delta",
    "uran_funo",
    "logsm_obs_ms",
    "logssfr_obs_ms",
    "logsm_obs_q",
    "logssfr_obs_q",
    "delta_scatter_ms",
    "delta_scatter_q",
    "t_table",
)
MCPhotInfo = namedtuple("MCPhotInfo", MCPHOT_INFO_KEYS)
