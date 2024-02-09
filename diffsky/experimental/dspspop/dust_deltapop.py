"""
"""

from collections import OrderedDict

import numpy as np
from jax import jit as jjit
from jax import lax, vmap

from ...utils import _sigmoid

LGSM_K = 5.0
LGSSFR_K = 5.0

DEFAULT_DUST_DELTA_PDICT = OrderedDict(
    dust_delta_logsm_x0_x0=10.0,
    dust_delta_logsm_x0_q=10.5,
    dust_delta_logsm_x0_ms=9.5,
    dust_delta_logsm_ylo_x0=-10.25,
    dust_delta_logsm_ylo_q=-0.5,
    dust_delta_logsm_ylo_ms=-0.2,
    dust_delta_logsm_yhi_x0=-11.25,
    dust_delta_logsm_yhi_q=-0.3,
    dust_delta_logsm_yhi_ms=-0.4,
)

LGSM_X0_BOUNDS = (8.0, 12.0)
LGSSFR_X0_BOUNDS = (-13.0, -7.0)
DUST_DELTA_BOUNDS = (-1.1, 0.4)
DUST_DELTA_BOUNDS_PDICT = OrderedDict(
    dust_delta_logsm_x0_x0=LGSM_X0_BOUNDS,
    dust_delta_logsm_x0_q=LGSM_X0_BOUNDS,
    dust_delta_logsm_x0_ms=LGSM_X0_BOUNDS,
    dust_delta_logsm_ylo_x0=LGSSFR_X0_BOUNDS,
    dust_delta_logsm_ylo_q=DUST_DELTA_BOUNDS,
    dust_delta_logsm_ylo_ms=DUST_DELTA_BOUNDS,
    dust_delta_logsm_yhi_x0=LGSSFR_X0_BOUNDS,
    dust_delta_logsm_yhi_q=DUST_DELTA_BOUNDS,
    dust_delta_logsm_yhi_ms=DUST_DELTA_BOUNDS,
)

DEFAULT_DUST_DELTA_PARAMS = np.array(list(DEFAULT_DUST_DELTA_PDICT.values()))
DUST_DELTA_PBOUNDS = np.array(list(DUST_DELTA_BOUNDS_PDICT.values()))


@jjit
def _get_dust_delta_galpop_from_u_params(
    gal_logsm, gal_logssfr, dust_delta_pop_u_params
):
    dust_delta_pop_params = _get_bounded_dust_delta_params(dust_delta_pop_u_params)
    dust_delta = _get_dust_delta_galpop_from_params(
        gal_logsm, gal_logssfr, dust_delta_pop_params
    )
    return dust_delta


@jjit
def _get_dust_delta_galpop_from_params(gal_logsm, gal_logssfr, dust_delta_pop_params):
    (
        dust_delta_logsm_x0_x0,
        dust_delta_logsm_x0_q,
        dust_delta_logsm_x0_ms,
        dust_delta_logsm_ylo_x0,
        dust_delta_logsm_ylo_q,
        dust_delta_logsm_ylo_ms,
        dust_delta_logsm_yhi_x0,
        dust_delta_logsm_yhi_q,
        dust_delta_logsm_yhi_ms,
    ) = dust_delta_pop_params

    dust_delta_logssfr_x0 = _sigmoid(
        gal_logsm,
        dust_delta_logsm_x0_x0,
        LGSM_K,
        dust_delta_logsm_ylo_x0,
        dust_delta_logsm_yhi_x0,
    )
    dust_delta_logssfr_q = _sigmoid(
        gal_logsm,
        dust_delta_logsm_x0_q,
        LGSM_K,
        dust_delta_logsm_ylo_q,
        dust_delta_logsm_yhi_q,
    )
    dust_delta_logssfr_ms = _sigmoid(
        gal_logsm,
        dust_delta_logsm_x0_ms,
        LGSSFR_K,
        dust_delta_logsm_ylo_ms,
        dust_delta_logsm_yhi_ms,
    )

    dust_delta = _sigmoid(
        gal_logssfr,
        dust_delta_logssfr_x0,
        LGSSFR_K,
        dust_delta_logssfr_q,
        dust_delta_logssfr_ms,
    )
    return dust_delta


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - lax.log(lnarg) / k


@jjit
def _get_bounded_dust_delta_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_dust_delta_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_dust_delta_params_kern = jjit(vmap(_get_bounded_dust_delta_param, in_axes=_C))
_get_dust_delta_u_params_kern = jjit(vmap(_get_unbounded_dust_delta_param, in_axes=_C))


@jjit
def _get_bounded_dust_delta_params(u_params):
    params = _get_dust_delta_params_kern(u_params, DUST_DELTA_PBOUNDS)
    return params


@jjit
def _get_unbounded_dust_delta_params(params):
    u_params = _get_dust_delta_u_params_kern(params, DUST_DELTA_PBOUNDS)
    return u_params


DEFAULT_DUST_DELTA_U_PARAMS = np.array(
    _get_unbounded_dust_delta_params(DEFAULT_DUST_DELTA_PARAMS)
)
