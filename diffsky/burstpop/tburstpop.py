"""
"""
from collections import OrderedDict, namedtuple

from dsps.sfh.diffburst import (
    DEFAULT_BURST_U_PARAMS,
    _get_tburst_params_from_tburst_u_params,
)
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

LGSM_K = 5.0
LGSSFR_K = 5.0
DEFAULT_U_LGYR_PEAK = DEFAULT_BURST_U_PARAMS.u_lgyr_peak
DEFAULT_U_LGYR_MAX = DEFAULT_BURST_U_PARAMS.u_lgyr_max

DEFAULT_TBURSTPOP_PDICT = OrderedDict(
    tburst_u_lgyr_peak_x0_logsm_x0=10.0,
    tburst_u_lgyr_peak_q_logsm_x0=-11.0,
    tburst_u_lgyr_peak_ms_logsm_x0=-11.0,
    tburst_u_lgyr_peak_x0_logsm_ylo=10.0,
    tburst_u_lgyr_peak_q_logsm_ylo=DEFAULT_U_LGYR_PEAK - 10,
    tburst_u_lgyr_peak_ms_logsm_ylo=DEFAULT_U_LGYR_PEAK + 10,
    tburst_u_lgyr_peak_x0_logsm_yhi=10.0,
    tburst_u_lgyr_peak_q_logsm_yhi=DEFAULT_U_LGYR_PEAK + 10,
    tburst_u_lgyr_peak_ms_logsm_yhi=DEFAULT_U_LGYR_PEAK - 10,
    tburst_u_lgyr_max_logsm_x0=10.0,
    tburst_u_lgyr_max_logsm_ylo=DEFAULT_U_LGYR_MAX - 10,
    tburst_u_lgyr_max_logsm_yhi=DEFAULT_U_LGYR_MAX + 10,
)
U_LGYR_PEAK_BOUNDS = (-2500, +2500)
U_LGYR_MAX_BOUNDS = (-2500, +2500)

DEFAULT_TBURSTPOP_BOUNDS_PDICT = OrderedDict(
    tburst_u_lgyr_peak_x0_logsm_x0=(8, 12.0),
    tburst_u_lgyr_peak_q_logsm_x0=(-13.0, -8.0),
    tburst_u_lgyr_peak_ms_logsm_x0=(-13.0, -8.0),
    tburst_u_lgyr_peak_x0_logsm_ylo=(8.0, 12.0),
    tburst_u_lgyr_peak_q_logsm_ylo=U_LGYR_PEAK_BOUNDS,
    tburst_u_lgyr_peak_ms_logsm_ylo=U_LGYR_PEAK_BOUNDS,
    tburst_u_lgyr_peak_x0_logsm_yhi=(8, 12.0),
    tburst_u_lgyr_peak_q_logsm_yhi=U_LGYR_PEAK_BOUNDS,
    tburst_u_lgyr_peak_ms_logsm_yhi=U_LGYR_PEAK_BOUNDS,
    tburst_u_lgyr_max_logsm_x0=(8.0, 12.0),
    tburst_u_lgyr_max_logsm_ylo=U_LGYR_MAX_BOUNDS,
    tburst_u_lgyr_max_logsm_yhi=U_LGYR_MAX_BOUNDS,
)

TburstPopParams = namedtuple("TburstPopParams", DEFAULT_TBURSTPOP_PDICT.keys())
_TBURSTPOP_UPNAMES = ["u_" + key for key in DEFAULT_TBURSTPOP_PDICT.keys()]
TburstPopUParams = namedtuple("TburstPopUParams", _TBURSTPOP_UPNAMES)


DEFAULT_TBURSTPOP_PARAMS = TburstPopParams(**DEFAULT_TBURSTPOP_PDICT)
TBURSTPOP_PBOUNDS = TburstPopParams(**DEFAULT_TBURSTPOP_BOUNDS_PDICT)


@jjit
def get_tburst_u_params_from_tburstpop_params(tburstpop_params, logsm, logssfr):
    u_lgyr_peak_x0 = _sigmoid(
        logsm,
        tburstpop_params.tburst_u_lgyr_peak_x0_logsm_x0,
        LGSM_K,
        tburstpop_params.tburst_u_lgyr_peak_q_logsm_x0,
        tburstpop_params.tburst_u_lgyr_peak_ms_logsm_x0,
    )
    u_lgyr_peak_ylo = _sigmoid(
        logsm,
        tburstpop_params.tburst_u_lgyr_peak_x0_logsm_ylo,
        LGSM_K,
        tburstpop_params.tburst_u_lgyr_peak_q_logsm_ylo,
        tburstpop_params.tburst_u_lgyr_peak_ms_logsm_ylo,
    )
    u_lgyr_peak_yhi = _sigmoid(
        logsm,
        tburstpop_params.tburst_u_lgyr_peak_x0_logsm_yhi,
        LGSM_K,
        tburstpop_params.tburst_u_lgyr_peak_q_logsm_yhi,
        tburstpop_params.tburst_u_lgyr_peak_ms_logsm_yhi,
    )

    u_lgyr_peak = _sigmoid(
        logssfr, u_lgyr_peak_x0, LGSSFR_K, u_lgyr_peak_ylo, u_lgyr_peak_yhi
    )

    u_lgyr_max = _sigmoid(
        logsm,
        tburstpop_params.tburst_u_lgyr_max_logsm_x0,
        LGSM_K,
        tburstpop_params.tburst_u_lgyr_max_logsm_ylo,
        tburstpop_params.tburst_u_lgyr_max_logsm_yhi,
    )

    tburst_u_params = u_lgyr_peak, u_lgyr_max
    return tburst_u_params


@jjit
def get_tburst_u_params_from_tburstpop_u_params(tburstpop_u_params, logsm, logssfr):
    tburstpop_params = get_bounded_tburstpop_params(tburstpop_u_params)
    u_lgyr_peak, u_lgyr_max = get_tburst_u_params_from_tburstpop_params(
        tburstpop_params, logsm, logssfr
    )
    tburst_u_params = u_lgyr_peak, u_lgyr_max
    return tburst_u_params


@jjit
def get_tburst_params_from_tburstpop_u_params(tburstpop_u_params, logsm, logssfr):
    tburst_u_params = get_tburst_u_params_from_tburstpop_u_params(
        tburstpop_u_params, logsm, logssfr
    )
    tburst_params = _get_tburst_params_from_tburst_u_params(*tburst_u_params)
    # lgyr_peak, lgyr_max = tburst_params
    return tburst_params


@jjit
def get_tburst_params_from_tburstpop_params(tburstpop_params, logsm, logssfr):
    tburst_u_params = get_tburst_u_params_from_tburstpop_params(
        tburstpop_params, logsm, logssfr
    )
    tburst_params = _get_tburst_params_from_tburst_u_params(*tburst_u_params)
    # lgyr_peak, lgyr_max = tburst_params
    return tburst_params


@jjit
def _get_bounded_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_tburstpop_params_kern = jjit(vmap(_get_bounded_param, in_axes=_C))
_get_tburstpop_u_params_kern = jjit(vmap(_get_unbounded_param, in_axes=_C))


@jjit
def get_bounded_tburstpop_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _TBURSTPOP_UPNAMES])
    params = _get_tburstpop_params_kern(
        jnp.array(u_params), jnp.array(TBURSTPOP_PBOUNDS)
    )
    return TburstPopParams(*params)


@jjit
def get_unbounded_tburstpop_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_TBURSTPOP_PARAMS._fields]
    )
    u_params = _get_tburstpop_u_params_kern(
        jnp.array(params), jnp.array(TBURSTPOP_PBOUNDS)
    )
    return TburstPopUParams(*u_params)


DEFAULT_TBURSTPOP_U_PARAMS = TburstPopUParams(
    *get_unbounded_tburstpop_params(DEFAULT_TBURSTPOP_PARAMS)
)
