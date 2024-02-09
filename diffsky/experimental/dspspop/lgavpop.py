"""
"""

from collections import OrderedDict

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ...utils import _sigmoid

LGSM_K = 5.0
LGSSFR_K = 5.0

DEFAULT_LGAV_PDICT = OrderedDict(
    lgav_logsm_x0_x0=10.0,
    lgav_logsm_x0_q=10.5,
    lgav_logsm_x0_ms=9.5,
    lgav_logsm_ylo_x0=-10.25,
    lgav_logsm_ylo_q=-2.0,
    lgav_logsm_ylo_ms=-0.75,
    lgav_logsm_yhi_x0=-11.25,
    lgav_logsm_yhi_q=-3.0,
    lgav_logsm_yhi_ms=-1.0,
)

LGSM_X0_BOUNDS = (8.0, 12.0)
LGSSFR_X0_BOUNDS = (-13.0, -7.0)
LGAV_BOUNDS = (-4.0, 1.5)
LGAV_BOUNDS_PDICT = OrderedDict(
    lgav_logsm_x0_x0=LGSM_X0_BOUNDS,
    lgav_logsm_x0_q=LGSM_X0_BOUNDS,
    lgav_logsm_x0_ms=LGSM_X0_BOUNDS,
    lgav_logsm_ylo_x0=LGSSFR_X0_BOUNDS,
    lgav_logsm_ylo_q=LGAV_BOUNDS,
    lgav_logsm_ylo_ms=LGAV_BOUNDS,
    lgav_logsm_yhi_x0=LGSSFR_X0_BOUNDS,
    lgav_logsm_yhi_q=LGAV_BOUNDS,
    lgav_logsm_yhi_ms=LGAV_BOUNDS,
)

DEFAULT_LGAV_PARAMS = np.array(list(DEFAULT_LGAV_PDICT.values()))
LGAV_PBOUNDS = np.array(list(LGAV_BOUNDS_PDICT.values()))


@jjit
def _get_lgav_galpop_from_u_params(gal_logsm, gal_logssfr, lgav_pop_u_params):
    lgav_pop_params = _get_bounded_lgav_params(lgav_pop_u_params)
    lgav = _get_lgav_galpop_from_params(gal_logsm, gal_logssfr, lgav_pop_params)
    return lgav


@jjit
def _get_lgav_galpop_from_params(gal_logsm, gal_logssfr, lgav_pop_params):
    (
        lgav_logsm_x0_x0,
        lgav_logsm_x0_q,
        lgav_logsm_x0_ms,
        lgav_logsm_ylo_x0,
        lgav_logsm_ylo_q,
        lgav_logsm_ylo_ms,
        lgav_logsm_yhi_x0,
        lgav_logsm_yhi_q,
        lgav_logsm_yhi_ms,
    ) = lgav_pop_params

    lgav_logssfr_x0 = _sigmoid(
        gal_logsm,
        lgav_logsm_x0_x0,
        LGSM_K,
        lgav_logsm_ylo_x0,
        lgav_logsm_yhi_x0,
    )
    lgav_logssfr_q = _sigmoid(
        gal_logsm,
        lgav_logsm_x0_q,
        LGSM_K,
        lgav_logsm_ylo_q,
        lgav_logsm_yhi_q,
    )
    lgav_logssfr_ms = _sigmoid(
        gal_logsm,
        lgav_logsm_x0_ms,
        LGSSFR_K,
        lgav_logsm_ylo_ms,
        lgav_logsm_yhi_ms,
    )

    lgav = _sigmoid(
        gal_logssfr,
        lgav_logssfr_x0,
        LGSSFR_K,
        lgav_logssfr_q,
        lgav_logssfr_ms,
    )
    return lgav


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def _get_bounded_lgav_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_lgav_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_lgav_params_kern = jjit(vmap(_get_bounded_lgav_param, in_axes=_C))
_get_lgav_u_params_kern = jjit(vmap(_get_unbounded_lgav_param, in_axes=_C))


@jjit
def _get_bounded_lgav_params(u_params):
    params = _get_lgav_params_kern(u_params, LGAV_PBOUNDS)
    return params


@jjit
def _get_unbounded_lgav_params(params):
    u_params = _get_lgav_u_params_kern(params, LGAV_PBOUNDS)
    return u_params


DEFAULT_LGAV_U_PARAMS = np.array(_get_unbounded_lgav_params(DEFAULT_LGAV_PARAMS))
