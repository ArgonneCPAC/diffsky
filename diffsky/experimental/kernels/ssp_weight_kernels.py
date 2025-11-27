""""""

from collections import namedtuple

from diffstar.diffstarpop.param_utils import mc_select_diffstar_params
from dsps.metallicity import umzr
from dsps.sed import metallicity_weights as zmetw
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ...burstpop import diffqburstpop_mono, freqburst_mono
from ...dustpop import tw_dustpop_mono_noise
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
def get_smooth_age_weights(diffstar_galpop, ssp_data, t_obs):
    age_weights_ms = calc_age_weights_from_sfh_table_vmap(
        diffstar_galpop.t_table, diffstar_galpop.sfh_ms, ssp_data.ssp_lg_age_gyr, t_obs
    )
    age_weights_q = calc_age_weights_from_sfh_table_vmap(
        diffstar_galpop.t_table, diffstar_galpop.sfh_q, ssp_data.ssp_lg_age_gyr, t_obs
    )
    return AgeWeights(ms=age_weights_ms, q=age_weights_q)


@jjit
def get_lgmet_weights(diffstar_galpop, ssp_data, t_obs, mzr_params, lgmet_scatter):
    # Calculate mean metallicity of the population
    lgmet_med_ms = umzr.mzr_model(diffstar_galpop.logsm_obs_ms, t_obs, *mzr_params)
    lgmet_med_q = umzr.mzr_model(diffstar_galpop.logsm_obs_q, t_obs, *mzr_params)

    # Compute metallicity distribution function
    lgmet_weights_ms = _calc_lgmet_weights_galpop(
        lgmet_med_ms, lgmet_scatter, ssp_data.ssp_lgmet
    )
    lgmet_weights_q = _calc_lgmet_weights_galpop(
        lgmet_med_q, lgmet_scatter, ssp_data.ssp_lgmet
    )
    return MetWeights(ms=lgmet_weights_ms, q=lgmet_weights_q)


@jjit
def get_smooth_ssp_weights(diffstar_galpop, ssp_data, t_obs, mzr_params, lgmet_scatter):
    age_weights = get_smooth_age_weights(diffstar_galpop, ssp_data, t_obs)
    lgmet_weights = get_lgmet_weights(
        diffstar_galpop, ssp_data, t_obs, mzr_params, lgmet_scatter
    )
    n_gals, n_age = age_weights.ms.shape
    n_met = lgmet_weights.ms.shape[1]

    weights_ms = combine_age_met_weights(age_weights.ms, lgmet_weights.ms)

    _w_age_q = age_weights.q.reshape((n_gals, 1, n_age))
    _w_lgmet_q = lgmet_weights.q.reshape((n_gals, n_met, 1))
    weights_q = _w_lgmet_q * _w_age_q
    weights_q = combine_age_met_weights(age_weights.q, lgmet_weights.q)

    weights = MSQ(ms=weights_ms, q=weights_q)

    return SSPWeights(
        weights=weights, age_weights=age_weights, lgmet_weights=lgmet_weights
    )


@jjit
def combine_age_met_weights(age_weights, lgmet_weights):
    n_gals, n_age = age_weights.shape
    n_met = lgmet_weights.shape[1]
    _w_age = age_weights.reshape((n_gals, 1, n_age))
    _w_lgmet = lgmet_weights.reshape((n_gals, n_met, 1))
    weights = _w_lgmet * _w_age
    return weights


@jjit
def compute_burstiness(diffstar_galpop, smooth_ssp_weights, ssp_data, burstpop_params):
    _args = (
        burstpop_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
        ssp_data.ssp_lg_age_gyr,
        smooth_ssp_weights.age_weights.ms,
    )
    _res = _calc_bursty_age_weights_vmap(*_args)
    bursty_age_weights_ms, burst_params = _res

    weights_bursty_ms = combine_age_met_weights(
        bursty_age_weights_ms, smooth_ssp_weights.lgmet_weights.ms
    )
    weights_smooth_q = combine_age_met_weights(
        smooth_ssp_weights.age_weights.q, smooth_ssp_weights.lgmet_weights.q
    )
    weights = MSQ(ms=weights_bursty_ms, q=weights_smooth_q)

    # burst_params = ('lgfburst', 'lgyr_peak', 'lgyr_max')

    # Calculate the frequency of SFH bursts
    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        burstpop_params.freqburst_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
    )
    return Burstiness(
        age_weights=bursty_age_weights_ms,
        weights=weights,
        burst_params=burst_params,
        p_burst=p_burst,
    )


@jjit
def compute_frac_ssp_errors(
    ssp_err_pop_params, z_obs, diffstar_galpop, wave_eff_galpop
):
    frac_ssp_err_ms = get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        diffstar_galpop.logsm_obs_ms,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )

    frac_ssp_err_q = get_frac_ssp_err_vmap(
        ssp_err_pop_params,
        z_obs,
        diffstar_galpop.logsm_obs_q,
        wave_eff_galpop,
        ssp_err_model.LAMBDA_REST,
    )
    return FracSSPErr(ms=frac_ssp_err_ms, q=frac_ssp_err_q)


@jjit
def compute_dust_attenuation(
    dust_key,
    diffstar_galpop,
    ssp_data,
    z_obs,
    wave_eff_galpop,
    dustpop_params,
    scatter_params,
):
    n_gals = z_obs.size
    # Generate randoms for stochasticity in dust attenuation curves
    av_key, delta_key, funo_key = jran.split(dust_key, 3)
    uran_av = jran.uniform(av_key, shape=(n_gals,))
    uran_delta = jran.uniform(delta_key, shape=(n_gals,))
    uran_funo = jran.uniform(funo_key, shape=(n_gals,))

    # Calculate fraction of flux transmitted through dust for each galaxy
    # Note that F_trans(λ_eff, τ_age) varies with stellar age τ_age
    ftrans_args_q = (
        dustpop_params,
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
    _res = calc_dust_ftrans_vmap(*ftrans_args_q)
    ftrans_q = _res[1]  # ftrans_q.shape = (n_gals, n_bands, n_age)
    noisy_dust_params_q = _res[3]  # fields = ('av', 'delta', 'funo')

    ftrans_args_ms = (
        dustpop_params,
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
    _res = calc_dust_ftrans_vmap(*ftrans_args_ms)
    ftrans_ms = _res[1]
    noisy_dust_params_ms = _res[3]  # fields = ('av', 'delta', 'funo')

    frac_trans = MSQ(ms=ftrans_ms, q=ftrans_q)
    dust_params = MSQ(ms=noisy_dust_params_ms, q=noisy_dust_params_q)
    dust_scatter = dust_params.q._replace(av=uran_av, delta=uran_delta, funo=uran_funo)

    return DustAttenuation(
        frac_trans=frac_trans, dust_params=dust_params, dust_scatter=dust_scatter
    )


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

    phot_info = PhotInfo(
        logmp_obs=diffstar_galpop.logmp_obs,
        logsm_obs=logsm_obs,
        logssfr_obs=logssfr_obs,
        sfh_table=sfh_table,
        obs_mags=mc_obs_mags,
        diffstar_params=diffstar_params,
        mc_sfh_type=mc_sfh_type,
        burstiness=burstiness,
        dust_params=dust_params,
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
    "diffstar_params",
    "mc_sfh_type",
    "burstiness",
    "dust_params",
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
PhotInfo = namedtuple("PhotInfo", MCPHOT_INFO_KEYS)
