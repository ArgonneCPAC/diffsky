"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

LGSM_K = 5.0
LGSSFR_K = 5.0

DEFAULT_FREQBURST_PDICT = OrderedDict(
    lgfreqburst_logsm_x0_x0=10.0,
    lgfreqburst_logsm_x0_q=10.5,
    lgfreqburst_logsm_x0_ms=9.5,
    lgfreqburst_logsm_ylo_x0=-10.25,
    lgfreqburst_logsm_ylo_q=-1.0,
    lgfreqburst_logsm_ylo_ms=-1.0,
    lgfreqburst_logsm_yhi_x0=-11.25,
    lgfreqburst_logsm_yhi_q=-1.0,
    lgfreqburst_logsm_yhi_ms=-1.0,
)

_LGSM_X0_BOUNDS = (8.0, 12.0)
_LGSSFR_X0_BOUNDS = (-13.0, -7.0)
_LGFREQBURST_BOUNDS = (-4.0, 0.0)
FREQBURST_BOUNDS_PDICT = OrderedDict(
    lgfreqburst_logsm_x0_x0=_LGSM_X0_BOUNDS,
    lgfreqburst_logsm_x0_q=_LGSM_X0_BOUNDS,
    lgfreqburst_logsm_x0_ms=_LGSM_X0_BOUNDS,
    lgfreqburst_logsm_ylo_x0=_LGSSFR_X0_BOUNDS,
    lgfreqburst_logsm_ylo_q=_LGFREQBURST_BOUNDS,
    lgfreqburst_logsm_ylo_ms=_LGFREQBURST_BOUNDS,
    lgfreqburst_logsm_yhi_x0=_LGSSFR_X0_BOUNDS,
    lgfreqburst_logsm_yhi_q=_LGFREQBURST_BOUNDS,
    lgfreqburst_logsm_yhi_ms=_LGFREQBURST_BOUNDS,
)

FreqburstParams = namedtuple("FreqburstParams", DEFAULT_FREQBURST_PDICT.keys())
_FREQBURST_UPNAMES = ["u_" + key for key in DEFAULT_FREQBURST_PDICT.keys()]
FreqburstUParams = namedtuple("FreqburstUParams", _FREQBURST_UPNAMES)


DEFAULT_FREQBURST_PARAMS = FreqburstParams(**DEFAULT_FREQBURST_PDICT)
FREQBURST_PBOUNDS = FreqburstParams(**FREQBURST_BOUNDS_PDICT)


@jjit
def get_lgfreqburst_from_freqburst_u_params(freqburst_u_params, logsm, logssfr):
    freqburst_params = get_bounded_freqburst_params(freqburst_u_params)
    lgfreqburst = get_lgfreqburst_from_freqburst_params(
        freqburst_params, logsm, logssfr
    )
    return lgfreqburst


@jjit
def get_lgfreqburst_from_freqburst_params(freqburst_params, logsm, logssfr):
    lgfreqburst_logssfr_x0 = _sigmoid(
        logsm,
        freqburst_params.lgfreqburst_logsm_x0_x0,
        LGSM_K,
        freqburst_params.lgfreqburst_logsm_ylo_x0,
        freqburst_params.lgfreqburst_logsm_yhi_x0,
    )
    lgfreqburst_logssfr_q = _sigmoid(
        logsm,
        freqburst_params.lgfreqburst_logsm_x0_q,
        LGSM_K,
        freqburst_params.lgfreqburst_logsm_ylo_q,
        freqburst_params.lgfreqburst_logsm_yhi_q,
    )
    lgfreqburst_logssfr_ms = _sigmoid(
        logsm,
        freqburst_params.lgfreqburst_logsm_x0_ms,
        LGSSFR_K,
        freqburst_params.lgfreqburst_logsm_ylo_ms,
        freqburst_params.lgfreqburst_logsm_yhi_ms,
    )

    lgfreqburst = _sigmoid(
        logssfr,
        lgfreqburst_logssfr_x0,
        LGSSFR_K,
        lgfreqburst_logssfr_q,
        lgfreqburst_logssfr_ms,
    )
    return lgfreqburst


@jjit
def _get_bounded_freqburst_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_freqburst_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_bounded_freqburst_params_kern = jjit(
    vmap(_get_bounded_freqburst_param, in_axes=_C)
)
_get_unbounded_freqburst_params_kern = jjit(
    vmap(_get_unbounded_freqburst_param, in_axes=_C)
)


@jjit
def get_bounded_freqburst_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _FREQBURST_UPNAMES])
    params = _get_bounded_freqburst_params_kern(
        jnp.array(u_params), jnp.array(FREQBURST_PBOUNDS)
    )
    freqburst_params = FreqburstParams(*params)
    return freqburst_params


@jjit
def get_unbounded_freqburst_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_FREQBURST_PARAMS._fields]
    )
    u_params = _get_unbounded_freqburst_params_kern(
        jnp.array(params), jnp.array(FREQBURST_PBOUNDS)
    )
    freqburst_u_params = FreqburstUParams(*u_params)
    return freqburst_u_params


DEFAULT_FREQBURST_U_PARAMS = FreqburstUParams(
    *get_unbounded_freqburst_params(DEFAULT_FREQBURST_PARAMS)
)
