"""
"""

from collections import namedtuple

from dsps.sfh.diffburst import BurstParams, calc_bursty_age_weights
from jax import jit as jjit

from .fburstpop import (
    DEFAULT_FBURSTPOP_PARAMS,
    get_bounded_fburstpop_params,
    get_lgfburst_from_fburstpop_params,
    get_unbounded_fburstpop_params,
)
from .freqburst import (
    DEFAULT_FREQBURST_PARAMS,
    get_bounded_freqburst_params,
    get_unbounded_freqburst_params,
)
from .tburstpop import (
    DEFAULT_TBURSTPOP_PARAMS,
    get_bounded_tburstpop_params,
    get_tburst_params_from_tburstpop_params,
    get_unbounded_tburstpop_params,
)

DiffburstPopParams = namedtuple(
    "DiffburstPopParams", ["freqburst_params", "fburstpop_params", "tburstpop_params"]
)
DEFAULT_DIFFBURSTPOP_PARAMS = DiffburstPopParams(
    DEFAULT_FREQBURST_PARAMS, DEFAULT_FBURSTPOP_PARAMS, DEFAULT_TBURSTPOP_PARAMS
)
_BURSTPOP_UPNAMES = [
    key.replace("params", "u_params") for key in DEFAULT_DIFFBURSTPOP_PARAMS._fields
]
DiffburstPopUParams = namedtuple("DiffburstPopUParams", _BURSTPOP_UPNAMES)


@jjit
def get_bounded_diffburstpop_params(u_params):
    u_freqburst_params, u_fburstpop_params, u_tburstpop_params = u_params
    bounded_freqburst_params = get_bounded_freqburst_params(u_freqburst_params)
    bounded_tburstpop_params = get_bounded_tburstpop_params(u_tburstpop_params)
    bounded_fburstpop_params = get_bounded_fburstpop_params(u_fburstpop_params)
    diffburstpop_params = DiffburstPopParams(
        bounded_freqburst_params, bounded_fburstpop_params, bounded_tburstpop_params
    )
    return diffburstpop_params


@jjit
def get_unbounded_diffburstpop_params(params):
    freqburst_params, fburstpop_params, tburstpop_params = params
    unbounded_freqburst_params = get_unbounded_freqburst_params(freqburst_params)
    unbounded_fburstpop_params = get_unbounded_fburstpop_params(fburstpop_params)
    unbounded_tburstpop_params = get_unbounded_tburstpop_params(tburstpop_params)
    diffburstpop_u_params = DiffburstPopUParams(
        unbounded_freqburst_params,
        unbounded_fburstpop_params,
        unbounded_tburstpop_params,
    )
    return diffburstpop_u_params


@jjit
def calc_bursty_age_weights_from_diffburstpop_params(
    diffburstpop_params, logsm, logssfr, ssp_lg_age_gyr, smooth_age_weights
):
    lgfburst = get_lgfburst_from_fburstpop_params(
        diffburstpop_params.fburstpop_params, logsm, logssfr
    )

    tburst_params = get_tburst_params_from_tburstpop_params(
        diffburstpop_params.tburstpop_params, logsm, logssfr
    )
    lgyr_peak, lgyr_max = tburst_params
    burst_params = BurstParams(lgfburst, lgyr_peak, lgyr_max)

    age_weights = calc_bursty_age_weights(
        burst_params, smooth_age_weights, ssp_lg_age_gyr
    )

    return age_weights, burst_params


def calc_bursty_age_weights_from_diffburstpop_u_params(
    diffburstpop_u_params, logsm, logssfr, ssp_lg_age_gyr, smooth_age_weights
):
    diffburstpop_params = get_bounded_diffburstpop_params(diffburstpop_u_params)
    args = diffburstpop_params, logsm, logssfr, ssp_lg_age_gyr, smooth_age_weights
    age_weights, burst_params = calc_bursty_age_weights_from_diffburstpop_params(*args)
    return age_weights, burst_params


DEFAULT_DIFFBURSTPOP_U_PARAMS = DiffburstPopUParams(
    *get_unbounded_diffburstpop_params(DEFAULT_DIFFBURSTPOP_PARAMS)
)
