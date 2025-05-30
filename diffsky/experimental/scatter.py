"""
"""
from ..utils.utility_funcs import _inverse_sigmoid, _sigmoid

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap


DEFAULT_SCATTER_PDICT = OrderedDict(
    delta_scatter=4.98,
    av_scatter=1.63,
    fburst_scatter=4.98,
    lgmet_scatter=5.72,
    funo_scatter=4.97,
)

K_BOUNDS = (0.0, 10.0)
MZR_BOUNDS = (0.01, 0.5)
FBURST_BOUNDS = (0.0, 1.0)

SCATTER_PBOUNDS_PDICT = OrderedDict(
    delta_scatter=K_BOUNDS,
    av_scatter=K_BOUNDS,
    fburst_scatter=K_BOUNDS,
    lgmet_scatter=K_BOUNDS,
    funo_scatter=K_BOUNDS,
)

ScatterParams = namedtuple("ScatterParams", DEFAULT_SCATTER_PDICT.keys())

_SCATTER_UPNAMES = ["u_" + key for key in SCATTER_PBOUNDS_PDICT.keys()]
ScatterUParams = namedtuple("ScatterUParams", _SCATTER_UPNAMES)

DEFAULT_SCATTER_PARAMS = ScatterParams(**DEFAULT_SCATTER_PDICT)
SCATTER_PBOUNDS = ScatterParams(**SCATTER_PBOUNDS_PDICT)


@jjit
def _get_bounded_scatter_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_scatter_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_scatter_params_kern = jjit(vmap(_get_bounded_scatter_param, in_axes=_C))
_get_scatter_u_params_kern = jjit(vmap(_get_unbounded_scatter_param, in_axes=_C))


@jjit
def get_bounded_scatter_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _SCATTER_UPNAMES])
    scatter_params = _get_scatter_params_kern(
        jnp.array(u_params), jnp.array(SCATTER_PBOUNDS)
    )
    return ScatterParams(*scatter_params)


@jjit
def get_unbounded_scatter_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_SCATTER_PARAMS._fields]
    )
    u_params = _get_scatter_u_params_kern(jnp.array(params), jnp.array(SCATTER_PBOUNDS))
    return ScatterUParams(*u_params)


DEFAULT_SCATTER_U_PARAMS = ScatterUParams(
    *get_unbounded_scatter_params(DEFAULT_SCATTER_PARAMS)
)
