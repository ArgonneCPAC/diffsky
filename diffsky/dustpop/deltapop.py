"""
"""
from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

LGSM_K = 5.0
LGSSFR_K = 5.0

DEFAULT_DELTAPOP_PDICT = OrderedDict(
    deltapop_logsm_x0_x0=10.0,
    deltapop_logsm_x0_q=10.5,
    deltapop_logsm_x0_ms=9.5,
    deltapop_logsm_ylo_x0=-10.25,
    deltapop_logsm_ylo_q=-0.5,
    deltapop_logsm_ylo_ms=-0.2,
    deltapop_logsm_yhi_x0=-11.25,
    deltapop_logsm_yhi_q=-0.3,
    deltapop_logsm_yhi_ms=-0.4,
)

LGSM_X0_BOUNDS = (8.0, 12.0)
LGSSFR_X0_BOUNDS = (-13.0, -7.0)
DELTAPOP_BOUNDS = (-1.1, 0.4)
DELTAPOP_BOUNDS_PDICT = OrderedDict(
    deltapop_logsm_x0_x0=LGSM_X0_BOUNDS,
    deltapop_logsm_x0_q=LGSM_X0_BOUNDS,
    deltapop_logsm_x0_ms=LGSM_X0_BOUNDS,
    deltapop_logsm_ylo_x0=LGSSFR_X0_BOUNDS,
    deltapop_logsm_ylo_q=DELTAPOP_BOUNDS,
    deltapop_logsm_ylo_ms=DELTAPOP_BOUNDS,
    deltapop_logsm_yhi_x0=LGSSFR_X0_BOUNDS,
    deltapop_logsm_yhi_q=DELTAPOP_BOUNDS,
    deltapop_logsm_yhi_ms=DELTAPOP_BOUNDS,
)


DeltaPopParams = namedtuple("DeltaPopParams", DEFAULT_DELTAPOP_PDICT.keys())
_DELTAPOP_UPNAMES = ["u_" + key for key in DEFAULT_DELTAPOP_PDICT.keys()]
DeltaPopUParams = namedtuple("DeltaPopUParams", _DELTAPOP_UPNAMES)

DEFAULT_DELTAPOP_PARAMS = DeltaPopParams(**DEFAULT_DELTAPOP_PDICT)
DELTAPOP_PBOUNDS = DeltaPopParams(**DELTAPOP_BOUNDS_PDICT)


@jjit
def get_delta_from_deltapop_u_params(deltapop_u_params, gal_logsm, gal_logssfr):
    deltapop_params = get_bounded_deltapop_params(deltapop_u_params)
    delta = get_delta_from_deltapop_params(deltapop_params, gal_logsm, gal_logssfr)
    return delta


@jjit
def get_delta_from_deltapop_params(deltapop_params, gal_logsm, gal_logssfr):
    deltapop_logssfr_x0 = _sigmoid(
        gal_logsm,
        deltapop_params.deltapop_logsm_x0_x0,
        LGSM_K,
        deltapop_params.deltapop_logsm_ylo_x0,
        deltapop_params.deltapop_logsm_yhi_x0,
    )
    deltapop_logssfr_q = _sigmoid(
        gal_logsm,
        deltapop_params.deltapop_logsm_x0_q,
        LGSM_K,
        deltapop_params.deltapop_logsm_ylo_q,
        deltapop_params.deltapop_logsm_yhi_q,
    )
    deltapop_logssfr_ms = _sigmoid(
        gal_logsm,
        deltapop_params.deltapop_logsm_x0_ms,
        LGSSFR_K,
        deltapop_params.deltapop_logsm_ylo_ms,
        deltapop_params.deltapop_logsm_yhi_ms,
    )

    deltapop = _sigmoid(
        gal_logssfr,
        deltapop_logssfr_x0,
        LGSSFR_K,
        deltapop_logssfr_q,
        deltapop_logssfr_ms,
    )
    return deltapop


@jjit
def _get_bounded_deltapop_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_deltapop_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_deltapop_params_kern = jjit(vmap(_get_bounded_deltapop_param, in_axes=_C))
_get_deltapop_u_params_kern = jjit(vmap(_get_unbounded_deltapop_param, in_axes=_C))


@jjit
def get_bounded_deltapop_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _DELTAPOP_UPNAMES])
    params = _get_deltapop_params_kern(u_params, jnp.array(DELTAPOP_PBOUNDS))
    return DeltaPopParams(*params)


@jjit
def get_unbounded_deltapop_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_DELTAPOP_PARAMS._fields]
    )
    u_params = _get_deltapop_u_params_kern(
        jnp.array(params), jnp.array(DELTAPOP_PBOUNDS)
    )
    return DeltaPopUParams(*u_params)


DEFAULT_DELTAPOP_U_PARAMS = DeltaPopUParams(
    *get_unbounded_deltapop_params(DEFAULT_DELTAPOP_PARAMS)
)
