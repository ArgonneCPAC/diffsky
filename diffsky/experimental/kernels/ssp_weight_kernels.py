""""""

from collections import namedtuple

from dsps.metallicity import umzr
from dsps.sed import metallicity_weights as zmetw
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import jit as jjit
from jax import vmap

_M = (0, None, None)
_calc_lgmet_weights_galpop = jjit(
    vmap(zmetw.calc_lgmet_weights_from_lognormal_mdf, in_axes=_M)
)

_AGEPOP = (None, 0, None, 0)
calc_age_weights_from_sfh_table_vmap = jjit(
    vmap(calc_age_weights_from_sfh_table, in_axes=_AGEPOP)
)

SmoothSSPWeights = namedtuple(
    "SmoothSSPWeights",
    (
        "smooth_age_weights_ms",
        "smooth_age_weights_q",
        "lgmet_weights_ms",
        "lgmet_weights_q",
    ),
)


@jjit
def get_smooth_ssp_weights(diffstar_galpop, ssp_data, t_obs, mzr_params, lgmet_scatter):
    # smooth_age_weights_ms.shape = (n_gals, n_age)
    smooth_age_weights_ms = calc_age_weights_from_sfh_table_vmap(
        diffstar_galpop.t_table, diffstar_galpop.sfh_ms, ssp_data.ssp_lg_age_gyr, t_obs
    )
    smooth_age_weights_q = calc_age_weights_from_sfh_table_vmap(
        diffstar_galpop.t_table, diffstar_galpop.sfh_q, ssp_data.ssp_lg_age_gyr, t_obs
    )

    # Calculate mean metallicity of the population
    lgmet_med_ms = umzr.mzr_model(diffstar_galpop.logsm_obs_ms, t_obs, *mzr_params)
    lgmet_med_q = umzr.mzr_model(diffstar_galpop.logsm_obs_q, t_obs, *mzr_params)

    # Calculate metallicity distribution function
    # lgmet_weights_q.shape = (n_gals, n_met)
    lgmet_weights_ms = _calc_lgmet_weights_galpop(
        lgmet_med_ms, lgmet_scatter, ssp_data.ssp_lgmet
    )
    lgmet_weights_q = _calc_lgmet_weights_galpop(
        lgmet_med_q, lgmet_scatter, ssp_data.ssp_lgmet
    )

    return SmoothSSPWeights(
        smooth_age_weights_ms=smooth_age_weights_ms,
        smooth_age_weights_q=smooth_age_weights_q,
        lgmet_weights_ms=lgmet_weights_ms,
        lgmet_weights_q=lgmet_weights_q,
    )
