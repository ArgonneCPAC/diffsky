"""
"""

from collections import OrderedDict, namedtuple
from copy import deepcopy

from jax import jit as jjit
from jax import nn
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

REDSHIFT_K = 5.0
LGSM_K = 5.0
LGSSFR_K = 5.0
LGAGE_K = 5.0

BOUNDING_K = 0.1

LGAGE_GYR_X0 = 6.5 - 9.0

DEFAULT_AVPOP_PDICT = OrderedDict(
    suav_logsm_x0=10.0,
    suav_logssfr_x0=-10.25,
    suav_logsm_ylo_q_z_ylo=-20.0,
    suav_logsm_ylo_ms_z_ylo=-20.0,
    suav_logsm_yhi_q_z_ylo=-20.0,
    suav_logsm_yhi_ms_z_ylo=-20.0,
    suav_logsm_ylo_q_z_yhi=-20.0,
    suav_logsm_ylo_ms_z_yhi=-20.0,
    suav_logsm_yhi_q_z_yhi=-20.0,
    suav_logsm_yhi_ms_z_yhi=-20.0,
    suav_z_x0=1.0,
    delta_suav_age=0.2,
)

LGSM_X0_BOUNDS = (9.0, 11.0)
LGSSFR_X0_BOUNDS = (-12.0, -8.0)
SUAV_BOUNDS = (-4.0, 1.5)
REDSHIFT_BOUNDS = (0.0, 5.0)
DELTA_SUAV_AGE_BOUNDS = (0.0, 1.0)
U_BOUNDS = (-100.0, 100.0)

AVPOP_PBOUNDS_PDICT = OrderedDict(
    suav_logsm_x0=LGSM_X0_BOUNDS,
    suav_logssfr_x0=LGSSFR_X0_BOUNDS,
    suav_logsm_ylo_q_z_ylo=U_BOUNDS,
    suav_logsm_ylo_ms_z_ylo=U_BOUNDS,
    suav_logsm_yhi_q_z_ylo=U_BOUNDS,
    suav_logsm_yhi_ms_z_ylo=U_BOUNDS,
    suav_logsm_ylo_q_z_yhi=U_BOUNDS,
    suav_logsm_ylo_ms_z_yhi=U_BOUNDS,
    suav_logsm_yhi_q_z_yhi=U_BOUNDS,
    suav_logsm_yhi_ms_z_yhi=U_BOUNDS,
    suav_z_x0=REDSHIFT_BOUNDS,
    delta_suav_age=DELTA_SUAV_AGE_BOUNDS,
)


AvPopParams = namedtuple("AvPopParams", DEFAULT_AVPOP_PDICT.keys())

_AVPOP_UPNAMES = ["u_" + key for key in AVPOP_PBOUNDS_PDICT.keys()]
AvPopUParams = namedtuple("AvPopUParams", _AVPOP_UPNAMES)


DEFAULT_AVPOP_PARAMS = AvPopParams(**DEFAULT_AVPOP_PDICT)
AVPOP_PBOUNDS = AvPopParams(**AVPOP_PBOUNDS_PDICT)

_EPS = 0.2
_EPS2 = 0.05
ZERODUST_AVPOP_PARAMS = deepcopy(DEFAULT_AVPOP_PARAMS)
ZERODUST_AVPOP_PARAMS = ZERODUST_AVPOP_PARAMS._replace(
    suav_logsm_ylo_q_z_ylo=U_BOUNDS[0] + _EPS,
    suav_logsm_ylo_ms_z_ylo=U_BOUNDS[0] + _EPS,
    suav_logsm_yhi_q_z_ylo=U_BOUNDS[0] + _EPS,
    suav_logsm_yhi_ms_z_ylo=U_BOUNDS[0] + _EPS,
    suav_logsm_ylo_q_z_yhi=U_BOUNDS[0] + _EPS,
    suav_logsm_ylo_ms_z_yhi=U_BOUNDS[0] + _EPS,
    suav_logsm_yhi_q_z_yhi=U_BOUNDS[0] + _EPS,
    suav_logsm_yhi_ms_z_yhi=U_BOUNDS[0] + _EPS,
    delta_suav_age=DELTA_SUAV_AGE_BOUNDS[0] + _EPS2,
)


@jjit
def double_sigmoid_monotonic(u_params, x, y, x0, y0, xk, yk, z_bounds):
    """4-D parameter space controlling double-sigmoid function z(x, y)
    such that ∂z/∂x>0 and ∂z/∂y>0 for all points (x, y)

    """
    u_x_lo_y_lo, u_x_lo_y_hi, u_x_hi_y_lo, u_x_hi_y_hi = u_params

    # (lo-x, lo-y) can be anything inside the z-bounds
    x_lo_y_lo = _sigmoid(u_x_lo_y_lo, 0.0, BOUNDING_K, *z_bounds)

    # (hi-x, lo-y) must be larger than (lo-x, lo-y)
    x_hi_y_lo = _sigmoid(u_x_hi_y_lo, 0.0, BOUNDING_K, x_lo_y_lo, z_bounds[1])

    # compute lower bound of z appropriate for the x-value
    ylo = _sigmoid(x, x0, xk, x_lo_y_lo, x_hi_y_lo)

    # Lowest value of z must be larger than ylo
    x_lo_y_hi = _sigmoid(u_x_lo_y_hi, 0.0, BOUNDING_K, ylo, z_bounds[1])

    # z must increase beyond its lowest value of x_lo_y_hi
    x_hi_y_hi = _sigmoid(u_x_hi_y_hi, 0.0, BOUNDING_K, x_lo_y_hi, z_bounds[1])

    # compute upper bound of z
    yhi = _sigmoid(x, x0, xk, x_lo_y_hi, x_hi_y_hi)

    z = _sigmoid(y, y0, yk, ylo, yhi)

    return z


@jjit
def get_av_from_avpop_params_scalar(avpop_params, logsm, logssfr, redshift, lg_age_gyr):

    DSM_ARGS = (
        avpop_params.suav_logsm_x0,
        avpop_params.suav_logssfr_x0,
        LGSM_K,
        LGSSFR_K,
        SUAV_BOUNDS,
    )
    params_z_ylo = (
        avpop_params.suav_logsm_ylo_q_z_ylo,
        avpop_params.suav_logsm_ylo_ms_z_ylo,
        avpop_params.suav_logsm_yhi_q_z_ylo,
        avpop_params.suav_logsm_yhi_ms_z_ylo,
    )
    suav_z_ylo = double_sigmoid_monotonic(params_z_ylo, logsm, logssfr, *DSM_ARGS)

    params_z_yhi = (
        avpop_params.suav_logsm_ylo_q_z_yhi,
        avpop_params.suav_logsm_ylo_ms_z_yhi,
        avpop_params.suav_logsm_yhi_q_z_yhi,
        avpop_params.suav_logsm_yhi_ms_z_yhi,
    )
    suav_z_yhi = double_sigmoid_monotonic(params_z_yhi, logsm, logssfr, *DSM_ARGS)

    suav = _sigmoid(
        redshift, avpop_params.suav_z_x0, REDSHIFT_K, suav_z_ylo, suav_z_yhi
    )

    delta_suav_age = _young_star_av_boost_kern(lg_age_gyr, avpop_params.delta_suav_age)
    suav = suav + delta_suav_age

    av = nn.softplus(suav)
    return av


@jjit
def _young_star_av_boost_kern(lg_age_gyr, u_av_boost_ostars):
    return _sigmoid(lg_age_gyr, LGAGE_GYR_X0, LGAGE_K, u_av_boost_ostars, 0)


@jjit
def _get_bounded_suav_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, BOUNDING_K, lo, hi)


@jjit
def _get_unbounded_suav_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, BOUNDING_K, lo, hi)


_C = (0, 0)
_get_avpop_params_kern = jjit(vmap(_get_bounded_suav_param, in_axes=_C))
_get_suav_u_params_kern = jjit(vmap(_get_unbounded_suav_param, in_axes=_C))


@jjit
def get_bounded_avpop_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _AVPOP_UPNAMES])
    avpop_params = _get_avpop_params_kern(jnp.array(u_params), jnp.array(AVPOP_PBOUNDS))
    return AvPopParams(*avpop_params)


@jjit
def get_unbounded_avpop_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_AVPOP_PARAMS._fields]
    )
    u_params = _get_suav_u_params_kern(jnp.array(params), jnp.array(AVPOP_PBOUNDS))
    return AvPopUParams(*u_params)


_AGE = (None, None, None, None, 0)
_POP = (None, 0, 0, 0, None)
get_av_from_avpop_params_galpop = jjit(
    vmap(vmap(get_av_from_avpop_params_scalar, in_axes=_AGE), in_axes=_POP)
)
get_av_from_avpop_params_singlegal = jjit(
    vmap(get_av_from_avpop_params_scalar, in_axes=_AGE)
)


@jjit
def get_av_from_avpop_u_params_singlegal(
    avpop_u_params, logsm, logssfr, redshift, lgage_gyr
):
    avpop_params = get_bounded_avpop_params(avpop_u_params)
    av = get_av_from_avpop_params_singlegal(
        avpop_params, logsm, logssfr, redshift, lgage_gyr
    )
    return av


@jjit
def get_av_from_avpop_u_params_galpop(
    avpop_u_params, logsm, logssfr, redshift, lgage_gyr
):
    avpop_params = get_bounded_avpop_params(avpop_u_params)
    av = get_av_from_avpop_params_galpop(
        avpop_params, logsm, logssfr, redshift, lgage_gyr
    )
    return av


DEFAULT_AVPOP_U_PARAMS = AvPopUParams(*get_unbounded_avpop_params(DEFAULT_AVPOP_PARAMS))
