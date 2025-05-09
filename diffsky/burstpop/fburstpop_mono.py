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

DEFAULT_FBURSTPOP_PDICT = OrderedDict(
    sufb_logsm_x0=10.0,
    sufb_logssfr_x0=-10.25,
    sufb_logsm_ylo_q=-10.0,
    sufb_logsm_ylo_ms=-10.0,
    sufb_logsm_yhi_q=-10.0,
    sufb_logsm_yhi_ms=-10.0,
)

LGSM_X0_BOUNDS = (8.0, 12.0)
LGSSFR_X0_BOUNDS = (-12.0, -8.0)
SUFB_BOUNDS = (-15.0, -2.0)

FBURSTPOP_PBOUNDS_PDICT = OrderedDict(
    sufb_logsm_x0=LGSM_X0_BOUNDS,
    sufb_logssfr_x0=LGSSFR_X0_BOUNDS,
    sufb_logsm_ylo_q=SUFB_BOUNDS,
    sufb_logsm_ylo_ms=SUFB_BOUNDS,
    sufb_logsm_yhi_q=SUFB_BOUNDS,
    sufb_logsm_yhi_ms=SUFB_BOUNDS,
)


FburstPopParams = namedtuple("FburstPopParams", DEFAULT_FBURSTPOP_PDICT.keys())

_FBURSTPOP_UPNAMES = ["u_" + key for key in FBURSTPOP_PBOUNDS_PDICT.keys()]
FburstPopUParams = namedtuple("FburstPopUParams", _FBURSTPOP_UPNAMES)


DEFAULT_FBURSTPOP_PARAMS = FburstPopParams(**DEFAULT_FBURSTPOP_PDICT)
FBURSTPOP_PBOUNDS = FburstPopParams(**FBURSTPOP_PBOUNDS_PDICT)

_EPS = 0.2
ZEROBURST_FBURSTPOP_PARAMS = deepcopy(DEFAULT_FBURSTPOP_PARAMS)
ZEROBURST_FBURSTPOP_PARAMS = ZEROBURST_FBURSTPOP_PARAMS._replace(
    sufb_logsm_ylo_q=SUFB_BOUNDS[0] + _EPS,
    sufb_logsm_ylo_ms=SUFB_BOUNDS[0] + _EPS,
    sufb_logsm_yhi_q=SUFB_BOUNDS[0] + _EPS,
    sufb_logsm_yhi_ms=SUFB_BOUNDS[0] + _EPS,
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
def get_fburst_from_fburstpop_params(fburstpop_params, logsm, logssfr):
    DSM_ARGS = (
        fburstpop_params.sufb_logsm_x0,
        fburstpop_params.sufb_logssfr_x0,
        LGSM_K,
        LGSSFR_K,
        SUFB_BOUNDS,
    )

    params = (
        fburstpop_params.sufb_logsm_ylo_q,
        fburstpop_params.sufb_logsm_ylo_ms,
        fburstpop_params.sufb_logsm_yhi_q,
        fburstpop_params.sufb_logsm_yhi_ms,
    )
    sufb = double_sigmoid_monotonic(params, logsm, logssfr, *DSM_ARGS)
    fburst = nn.softplus(sufb)
    return fburst


@jjit
def get_fburst_from_fburstpop_u_params(fburstpop_u_params, logsm, logssfr):
    fburstpop_params = get_bounded_fburstpop_params(fburstpop_u_params)
    fburst = get_fburst_from_fburstpop_params(fburstpop_params, logsm, logssfr)
    return fburst


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
