"""
"""

from collections import OrderedDict

import numpy as np
from jax import jit as jjit
from jax import lax, vmap

from ...utils import _sigmoid

LGSM_K = 5.0
LGSSFR_K = 5.0

DEFAULT_LGFBURST_PDICT = OrderedDict(
    lgfburst_logsm_x0_x0=10.0,
    lgfburst_logsm_x0_q=10.5,
    lgfburst_logsm_x0_ms=9.5,
    lgfburst_logsm_ylo_x0=-10.25,
    lgfburst_logsm_ylo_q=-3.0,
    lgfburst_logsm_ylo_ms=-2.0,
    lgfburst_logsm_yhi_x0=-11.25,
    lgfburst_logsm_yhi_q=-4.0,
    lgfburst_logsm_yhi_ms=-2.5,
)

LGSM_X0_BOUNDS = (8.0, 12.0)
LGSSFR_X0_BOUNDS = (-13.0, -7.0)
LGFBURST_BOUNDS = (-5.0, -1.0)
LGFBURST_BOUNDS_PDICT = OrderedDict(
    lgfburst_logsm_x0_x0=LGSM_X0_BOUNDS,
    lgfburst_logsm_x0_q=LGSM_X0_BOUNDS,
    lgfburst_logsm_x0_ms=LGSM_X0_BOUNDS,
    lgfburst_logsm_ylo_x0=LGSSFR_X0_BOUNDS,
    lgfburst_logsm_ylo_q=LGFBURST_BOUNDS,
    lgfburst_logsm_ylo_ms=LGFBURST_BOUNDS,
    lgfburst_logsm_yhi_x0=LGSSFR_X0_BOUNDS,
    lgfburst_logsm_yhi_q=LGFBURST_BOUNDS,
    lgfburst_logsm_yhi_ms=LGFBURST_BOUNDS,
)

DEFAULT_LGFBURST_PARAMS = np.array(list(DEFAULT_LGFBURST_PDICT.values()))
LGFBURST_PBOUNDS = np.array(list(LGFBURST_BOUNDS_PDICT.values()))


@jjit
def _get_lgfburst_galpop_from_u_params(gal_logsm, gal_logssfr, lgfburst_pop_u_params):
    lgfburst_pop_params = _get_bounded_lgfburst_params(lgfburst_pop_u_params)
    lgfburst = _get_lgfburst_galpop_from_params(
        gal_logsm, gal_logssfr, lgfburst_pop_params
    )
    return lgfburst


@jjit
def _get_lgfburst_galpop_from_params(gal_logsm, gal_logssfr, lgfburst_pop_params):
    (
        lgfburst_logsm_x0_x0,
        lgfburst_logsm_x0_q,
        lgfburst_logsm_x0_ms,
        lgfburst_logsm_ylo_x0,
        lgfburst_logsm_ylo_q,
        lgfburst_logsm_ylo_ms,
        lgfburst_logsm_yhi_x0,
        lgfburst_logsm_yhi_q,
        lgfburst_logsm_yhi_ms,
    ) = lgfburst_pop_params

    lgfburst_logssfr_x0 = _sigmoid(
        gal_logsm,
        lgfburst_logsm_x0_x0,
        LGSM_K,
        lgfburst_logsm_ylo_x0,
        lgfburst_logsm_yhi_x0,
    )
    lgfburst_logssfr_q = _sigmoid(
        gal_logsm,
        lgfburst_logsm_x0_q,
        LGSM_K,
        lgfburst_logsm_ylo_q,
        lgfburst_logsm_yhi_q,
    )
    lgfburst_logssfr_ms = _sigmoid(
        gal_logsm,
        lgfburst_logsm_x0_ms,
        LGSSFR_K,
        lgfburst_logsm_ylo_ms,
        lgfburst_logsm_yhi_ms,
    )

    lgfburst = _sigmoid(
        gal_logssfr,
        lgfburst_logssfr_x0,
        LGSSFR_K,
        lgfburst_logssfr_q,
        lgfburst_logssfr_ms,
    )
    return lgfburst


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - lax.log(lnarg) / k


@jjit
def _get_bounded_lgfburst_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_lgfburst_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_lgfburst_params_kern = jjit(vmap(_get_bounded_lgfburst_param, in_axes=_C))
_get_lgfburst_u_params_kern = jjit(vmap(_get_unbounded_lgfburst_param, in_axes=_C))


@jjit
def _get_bounded_lgfburst_params(u_params):
    params = _get_lgfburst_params_kern(u_params, LGFBURST_PBOUNDS)
    return params


@jjit
def _get_unbounded_lgfburst_params(params):
    u_params = _get_lgfburst_u_params_kern(params, LGFBURST_PBOUNDS)
    return u_params


DEFAULT_LGFBURST_U_PARAMS = np.array(
    _get_unbounded_lgfburst_params(DEFAULT_LGFBURST_PARAMS)
)
