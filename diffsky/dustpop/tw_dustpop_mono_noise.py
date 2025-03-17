""" """

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

DEFAULT_DUSTPOP_SCATTER_PDICT = OrderedDict(
    delta_scatter=5.0,
    av_scatter=5.0,
    lgfburst_scatter=5.0,
    lgmet_scatter=5.0,
    funo_scatter=5.0,
)

K_BOUNDS = (0.0, 10.0)

DUSTPOP_SCATTER_PBOUNDS_PDICT = OrderedDict(
    delta_scatter=K_BOUNDS,
    av_scatter=K_BOUNDS,
    lgfburst_scatter=K_BOUNDS,
    lgmet_scatter=K_BOUNDS,
    funo_scatter=K_BOUNDS,
)

DustScatterParams = namedtuple("ScatterParams", DEFAULT_DUSTPOP_SCATTER_PDICT.keys())

_SCATTER_UPNAMES = ["u_" + key for key in DUSTPOP_SCATTER_PBOUNDS_PDICT.keys()]
DustScatterUParams = namedtuple("DustScatterUParams", _SCATTER_UPNAMES)

DEFAULT_DUSTPOP_SCATTER_PARAMS = DustScatterParams(**DEFAULT_DUSTPOP_SCATTER_PDICT)
DUSTPOP_SCATTER_PBOUNDS = DustScatterParams(**DUSTPOP_SCATTER_PBOUNDS_PDICT)


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
def get_bounded_dustpop_scatter_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _SCATTER_UPNAMES])
    scatter_params = _get_scatter_params_kern(
        jnp.array(u_params), jnp.array(DUSTPOP_SCATTER_PBOUNDS)
    )
    return DustScatterParams(*scatter_params)


@jjit
def get_unbounded_dustpop_scatter_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_DUSTPOP_SCATTER_PARAMS._fields]
    )
    u_params = _get_scatter_u_params_kern(
        jnp.array(params), jnp.array(DUSTPOP_SCATTER_PBOUNDS)
    )
    return DustScatterUParams(*u_params)


DEFAULT_DUSTPOP_SCATTER_U_PARAMS = DustScatterUParams(
    *get_unbounded_dustpop_scatter_params(DEFAULT_DUSTPOP_SCATTER_PARAMS)
)
