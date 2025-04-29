""" """

from collections import OrderedDict, namedtuple
from copy import deepcopy

from dsps.utils import _inverse_sigmoid, _sigmoid
from jax import jit as jjit
from jax import nn
from jax import numpy as jnp
from jax import vmap

BOUNDING_K = 0.1
LGSM_K = 5.0
LGSSFR_K = 5.0

DEFAULT_FREQBURST_PDICT = OrderedDict(
    sufqb_logsm_x0=10.0,
    sufqb_logssfr_x0=-10.25,
    sufqb_logsm_ylo_q=-5.0,
    sufqb_logsm_ylo_ms=-0.2,
    sufqb_logsm_yhi_q=-5.0,
    sufqb_logsm_yhi_ms=-0.5,
)

LGSM_X0_BOUNDS = (9.0, 11.0)
LGSSFR_X0_BOUNDS = (-12.0, -8.0)
SUFQB_BOUNDS = (-10.0, 0.5)

FREQBURST_PBOUNDS_PDICT = OrderedDict(
    sufqb_logsm_x0=LGSM_X0_BOUNDS,
    sufqb_logssfr_x0=LGSSFR_X0_BOUNDS,
    sufqb_logsm_ylo_q=SUFQB_BOUNDS,
    sufqb_logsm_ylo_ms=SUFQB_BOUNDS,
    sufqb_logsm_yhi_q=SUFQB_BOUNDS,
    sufqb_logsm_yhi_ms=SUFQB_BOUNDS,
)


FreqburstParams = namedtuple("FreqburstParams", DEFAULT_FREQBURST_PDICT.keys())

_FREQBURST_UPNAMES = ["u_" + key for key in FREQBURST_PBOUNDS_PDICT.keys()]
FreqburstUParams = namedtuple("FreqburstUParams", _FREQBURST_UPNAMES)


DEFAULT_FREQBURST_PARAMS = FreqburstParams(**DEFAULT_FREQBURST_PDICT)
FREQBURST_PBOUNDS = FreqburstParams(**FREQBURST_PBOUNDS_PDICT)

_EPS = 0.2
ZEROBURST_FREQBURST_PARAMS = deepcopy(DEFAULT_FREQBURST_PARAMS)
ZEROBURST_FREQBURST_PARAMS = ZEROBURST_FREQBURST_PARAMS._replace(
    sufqb_logsm_ylo_q=SUFQB_BOUNDS[0] + _EPS,
    sufqb_logsm_ylo_ms=SUFQB_BOUNDS[0] + _EPS,
    sufqb_logsm_yhi_q=SUFQB_BOUNDS[0] + _EPS,
    sufqb_logsm_yhi_ms=SUFQB_BOUNDS[0] + _EPS,
)


@jjit
def double_sigmoid_monotonic(u_params, x, y, x0, y0, xk, yk, z_bounds):
    """4-D parameter space controlling double-sigmoid function z(x, y)
    such that ∂z/∂x<0 and ∂z/∂y>0 for all points (x, y)

    """
    u_x_lo_y_lo, u_x_lo_y_hi, u_x_hi_y_lo, u_x_hi_y_hi = u_params

    # (hi-x, lo-y) can be anything inside the z-bounds
    x_hi_y_lo = _sigmoid(u_x_hi_y_lo, 0.0, BOUNDING_K, *z_bounds)

    # (lo-x, lo-y) must be larger than (hi-x, lo-y)
    x_lo_y_lo = _sigmoid(u_x_lo_y_lo, 0.0, BOUNDING_K, x_hi_y_lo, z_bounds[1])

    # compute lower bound of z appropriate for the x-value
    ylo = _sigmoid(x, x0, xk, x_lo_y_lo, x_hi_y_lo)

    # Lowest value of z must be larger than ylo
    x_hi_y_hi = _sigmoid(u_x_hi_y_hi, 0.0, BOUNDING_K, ylo, z_bounds[1])

    # z must increase beyond its lowest value of x_hi_y_hi
    x_lo_y_hi = _sigmoid(u_x_lo_y_hi, 0.0, BOUNDING_K, x_hi_y_hi, z_bounds[1])

    # compute upper bound of z
    yhi = _sigmoid(x, x0, xk, x_lo_y_hi, x_hi_y_hi)

    z = _sigmoid(y, y0, yk, ylo, yhi)

    return z


@jjit
def get_freqburst_from_freqburst_params(freqburst_params, logsm, logssfr):
    DSM_ARGS = (
        freqburst_params.sufqb_logsm_x0,
        freqburst_params.sufqb_logssfr_x0,
        LGSM_K,
        LGSSFR_K,
        SUFQB_BOUNDS,
    )

    params = (
        freqburst_params.sufqb_logsm_ylo_q,
        freqburst_params.sufqb_logsm_ylo_ms,
        freqburst_params.sufqb_logsm_yhi_q,
        freqburst_params.sufqb_logsm_yhi_ms,
    )
    sufqb = double_sigmoid_monotonic(params, logsm, logssfr, *DSM_ARGS)
    freq_burst = nn.softplus(sufqb)
    return freq_burst


@jjit
def get_freqburst_from_freqburst_u_params(freqburst_u_params, logsm, logssfr):
    freqburst_params = get_bounded_freqburst_params(freqburst_u_params)
    freqburst = get_freqburst_from_freqburst_params(freqburst_params, logsm, logssfr)
    return freqburst


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
