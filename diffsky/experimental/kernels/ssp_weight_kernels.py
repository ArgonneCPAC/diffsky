""""""

from collections import namedtuple

from dsps.metallicity import umzr
from dsps.sed import metallicity_weights as zmetw
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import jit as jjit
from jax import vmap

from ...burstpop import diffqburstpop_mono, freqburst_mono

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

AgeWeights = namedtuple("AgeWeights", ("ms", "q"))
MetWeights = namedtuple("MetWeights", ("ms", "q"))
SSPWeights = namedtuple("SSPWeights", ("age_weights", "lgmet_weights"))
Burstiness = namedtuple("Burstiness", ("age_weights", "burst_params", "p_burst"))


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
    return SSPWeights(age_weights=age_weights, lgmet_weights=lgmet_weights)


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
    age_weights, burst_params = _res
    # bursty_age_weights_ms.shape = (n_gals, age)
    # burst_params = ('lgfburst', 'lgyr_peak', 'lgyr_max')

    # Calculate the frequency of SFH bursts
    p_burst = freqburst_mono.get_freqburst_from_freqburst_params(
        burstpop_params.freqburst_params,
        diffstar_galpop.logsm_obs_ms,
        diffstar_galpop.logssfr_obs_ms,
    )
    return Burstiness(
        age_weights=age_weights, burst_params=burst_params, p_burst=p_burst
    )
