"""
"""

from collections import OrderedDict, namedtuple
from copy import deepcopy

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

LGSM_K = 5.0
LGSSFR_K = 5.0

DEFAULT_FBURSTPOP_PDICT = OrderedDict(
    lgfburst_logsm_x0_x0=10.0,
    lgfburst_logsm_x0_q=10.5,
    lgfburst_logsm_x0_ms=9.5,
    lgfburst_logsm_ylo_x0=-10.25,
    lgfburst_logsm_ylo_q=-4.5,
    lgfburst_logsm_ylo_ms=-4.5,
    lgfburst_logsm_yhi_x0=-11.25,
    lgfburst_logsm_yhi_q=-4.5,
    lgfburst_logsm_yhi_ms=-4.5,
)

_LGSM_X0_BOUNDS = (8.0, 12.0)
_LGSSFR_X0_BOUNDS = (-13.0, -7.0)
_LGFBURST_BOUNDS = (-6.0, -2.0)
FBURSTPOP_BOUNDS_PDICT = OrderedDict(
    lgfburst_logsm_x0_x0=_LGSM_X0_BOUNDS,
    lgfburst_logsm_x0_q=_LGSM_X0_BOUNDS,
    lgfburst_logsm_x0_ms=_LGSM_X0_BOUNDS,
    lgfburst_logsm_ylo_x0=_LGSSFR_X0_BOUNDS,
    lgfburst_logsm_ylo_q=_LGFBURST_BOUNDS,
    lgfburst_logsm_ylo_ms=_LGFBURST_BOUNDS,
    lgfburst_logsm_yhi_x0=_LGSSFR_X0_BOUNDS,
    lgfburst_logsm_yhi_q=_LGFBURST_BOUNDS,
    lgfburst_logsm_yhi_ms=_LGFBURST_BOUNDS,
)

FburstPopParams = namedtuple("FburstPopParams", DEFAULT_FBURSTPOP_PDICT.keys())
_FBURSTPOP_UPNAMES = ["u_" + key for key in DEFAULT_FBURSTPOP_PDICT.keys()]
FburstPopUParams = namedtuple("FburstPopUParams", _FBURSTPOP_UPNAMES)


DEFAULT_FBURSTPOP_PARAMS = FburstPopParams(**DEFAULT_FBURSTPOP_PDICT)
FBURSTPOP_PBOUNDS = FburstPopParams(**FBURSTPOP_BOUNDS_PDICT)

_EPS = 0.1
ZEROBURST_FBURSTPOP_PARAMS = deepcopy(DEFAULT_FBURSTPOP_PARAMS)
ZEROBURST_FBURSTPOP_PARAMS = ZEROBURST_FBURSTPOP_PARAMS._replace(
    lgfburst_logsm_ylo_q=_LGFBURST_BOUNDS[0] + _EPS,
    lgfburst_logsm_ylo_ms=_LGFBURST_BOUNDS[0] + _EPS,
    lgfburst_logsm_yhi_q=_LGFBURST_BOUNDS[0] + _EPS,
    lgfburst_logsm_yhi_ms=_LGFBURST_BOUNDS[0] + _EPS,
)


@jjit
def get_lgfburst_from_fburstpop_u_params(fburstpop_u_params, logsm, logssfr):
    fburstpop_params = get_bounded_fburstpop_params(fburstpop_u_params)
    lgfburst = get_lgfburst_from_fburstpop_params(fburstpop_params, logsm, logssfr)
    return lgfburst


@jjit
def get_lgfburst_from_fburstpop_params(fburstpop_params, logsm, logssfr):
    lgfburst_logssfr_x0 = _sigmoid(
        logsm,
        fburstpop_params.lgfburst_logsm_x0_x0,
        LGSM_K,
        fburstpop_params.lgfburst_logsm_ylo_x0,
        fburstpop_params.lgfburst_logsm_yhi_x0,
    )
    lgfburst_logssfr_q = _sigmoid(
        logsm,
        fburstpop_params.lgfburst_logsm_x0_q,
        LGSM_K,
        fburstpop_params.lgfburst_logsm_ylo_q,
        fburstpop_params.lgfburst_logsm_yhi_q,
    )
    lgfburst_logssfr_ms = _sigmoid(
        logsm,
        fburstpop_params.lgfburst_logsm_x0_ms,
        LGSSFR_K,
        fburstpop_params.lgfburst_logsm_ylo_ms,
        fburstpop_params.lgfburst_logsm_yhi_ms,
    )

    lgfburst = _sigmoid(
        logssfr,
        lgfburst_logssfr_x0,
        LGSSFR_K,
        lgfburst_logssfr_q,
        lgfburst_logssfr_ms,
    )
    return lgfburst


@jjit
def _get_bounded_fburstpop_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_fburstpop_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_bounded_fburstpop_params_kern = jjit(
    vmap(_get_bounded_fburstpop_param, in_axes=_C)
)
_get_unbounded_fburstpop_params_kern = jjit(
    vmap(_get_unbounded_fburstpop_param, in_axes=_C)
)


@jjit
def get_bounded_fburstpop_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _FBURSTPOP_UPNAMES])
    params = _get_bounded_fburstpop_params_kern(
        jnp.array(u_params), jnp.array(FBURSTPOP_PBOUNDS)
    )
    fburstpop_params = FburstPopParams(*params)
    return fburstpop_params


@jjit
def get_unbounded_fburstpop_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_FBURSTPOP_PARAMS._fields]
    )
    u_params = _get_unbounded_fburstpop_params_kern(
        jnp.array(params), jnp.array(FBURSTPOP_PBOUNDS)
    )
    fburstpop_u_params = FburstPopUParams(*u_params)
    return fburstpop_u_params


DEFAULT_FBURSTPOP_U_PARAMS = FburstPopUParams(
    *get_unbounded_fburstpop_params(DEFAULT_FBURSTPOP_PARAMS)
)
