""" """

from collections import namedtuple

from dsps.sfh.diffburst import BurstParams, calc_bursty_age_weights
from jax import jit as jjit
from jax import numpy as jnp

from .fburstpop_mono import (
    DEFAULT_FBURSTPOP_PARAMS,
    ZEROBURST_FBURSTPOP_PARAMS,
    get_bounded_fburstpop_params,
    get_fburst_from_fburstpop_params,
    get_unbounded_fburstpop_params,
    _get_bounded_fburstpop_param,
    _get_unbounded_fburstpop_param
)
from .freqburst_mono import (
    DEFAULT_FREQBURST_PARAMS,
    ZEROBURST_FREQBURST_PARAMS,
    get_bounded_freqburst_params,
    get_unbounded_freqburst_params,
)
from .tburstpop import (
    DEFAULT_TBURSTPOP_PARAMS,
    get_bounded_tburstpop_params,
    get_tburst_params_from_tburstpop_params,
    get_unbounded_tburstpop_params,
)

from ..utils.utility_funcs import _inverse_sigmoid

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

ZERO_DIFFBURSTPOP_PARAMS = DiffburstPopParams(
    ZEROBURST_FREQBURST_PARAMS, ZEROBURST_FBURSTPOP_PARAMS, DEFAULT_TBURSTPOP_PARAMS
)

FBURST_BOUNDS = (0.0, 1.0)


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
    diffburstpop_params,
    logsm,
    logssfr,
    ssp_lg_age_gyr,
    smooth_age_weights,
    random_draw_burst,
    scatter_params,
):
    f_burst = get_fburst_from_fburstpop_params(
        diffburstpop_params.fburstpop_params, logsm, logssfr
    )
    ufburst = _get_unbounded_fburstpop_param(f_burst, FBURST_BOUNDS)
    noisy_ufburst = _inverse_sigmoid(random_draw_burst, ufburst, scatter_params.fburst_scatter, 0.0, 1.0)
    noisy_fburst = _get_bounded_fburstpop_param(noisy_ufburst, FBURST_BOUNDS)
    lgfburst = jnp.log10(noisy_fburst)

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
    diffburstpop_u_params,
    logsm,
    logssfr,
    ssp_lg_age_gyr,
    smooth_age_weights,
    random_draw_burst,
    scatter_params,

):
    diffburstpop_params = get_bounded_diffburstpop_params(diffburstpop_u_params)
    args = (
        diffburstpop_params,
        logsm,
        logssfr,
        ssp_lg_age_gyr,
        smooth_age_weights,
        random_draw_burst,
        scatter_params)
    age_weights, burst_params = calc_bursty_age_weights_from_diffburstpop_params(*args)
    return age_weights, burst_params


DEFAULT_DIFFBURSTPOP_U_PARAMS = DiffburstPopUParams(
    *get_unbounded_diffburstpop_params(DEFAULT_DIFFBURSTPOP_PARAMS)
)
ZERO_DIFFBURSTPOP_U_PARAMS = get_unbounded_diffburstpop_params(
    DEFAULT_DIFFBURSTPOP_PARAMS
)
