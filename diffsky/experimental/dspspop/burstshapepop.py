"""
"""

from collections import OrderedDict

import numpy as np
from dsps.sfh.diffburst import DEFAULT_BURST_U_PARAMS
from jax import jit as jjit
from jax import lax, vmap

from ...utils import _sigmoid

LGSM_K = 5.0
LGSSFR_K = 5.0


DEFAULT_U_LGYR_PEAK = DEFAULT_BURST_U_PARAMS.u_lgyr_peak
DEFAULT_U_LGYR_MAX = DEFAULT_BURST_U_PARAMS.u_lgyr_max

DEFAULT_BURSTSHAPE_PDICT = OrderedDict(
    burstshape_u_lgyr_peak_x0_logsm_x0=10.0,
    burstshape_u_lgyr_peak_q_logsm_x0=-11.0,
    burstshape_u_lgyr_peak_ms_logsm_x0=-11.0,
    burstshape_u_lgyr_peak_x0_logsm_ylo=10.0,
    burstshape_u_lgyr_peak_q_logsm_ylo=DEFAULT_U_LGYR_PEAK - 10,
    burstshape_u_lgyr_peak_ms_logsm_ylo=DEFAULT_U_LGYR_PEAK + 10,
    burstshape_u_lgyr_peak_x0_logsm_yhi=10.0,
    burstshape_u_lgyr_peak_q_logsm_yhi=DEFAULT_U_LGYR_PEAK + 10,
    burstshape_u_lgyr_peak_ms_logsm_yhi=DEFAULT_U_LGYR_PEAK - 10,
    burstshape_u_lgyr_max_logsm_x0=10.0,
    burstshape_u_lgyr_max_logsm_ylo=DEFAULT_U_LGYR_MAX - 10,
    burstshape_u_lgyr_max_logsm_yhi=DEFAULT_U_LGYR_MAX + 10,
)
U_LGYR_PEAK_BOUNDS = (-2500, +2500)
U_LGYR_MAX_BOUNDS = (-2500, +2500)
DEFAULT_BURSTSHAPE_BOUNDS_PDICT = OrderedDict(
    burstshape_u_lgyr_peak_x0_logsm_x0=(8, 12.0),
    burstshape_u_lgyr_peak_q_logsm_x0=(-13.0, -8.0),
    burstshape_u_lgyr_peak_ms_logsm_x0=(-13.0, -8.0),
    burstshape_u_lgyr_peak_x0_logsm_ylo=(8.0, 12.0),
    burstshape_u_lgyr_peak_q_logsm_ylo=U_LGYR_PEAK_BOUNDS,
    burstshape_u_lgyr_peak_ms_logsm_ylo=U_LGYR_PEAK_BOUNDS,
    burstshape_u_lgyr_peak_x0_logsm_yhi=(8, 12.0),
    burstshape_u_lgyr_peak_q_logsm_yhi=U_LGYR_PEAK_BOUNDS,
    burstshape_u_lgyr_peak_ms_logsm_yhi=U_LGYR_PEAK_BOUNDS,
    burstshape_u_lgyr_max_logsm_x0=(8.0, 12.0),
    burstshape_u_lgyr_max_logsm_ylo=U_LGYR_MAX_BOUNDS,
    burstshape_u_lgyr_max_logsm_yhi=U_LGYR_MAX_BOUNDS,
)

DEFAULT_BURSTSHAPE_PARAMS = np.array(list(DEFAULT_BURSTSHAPE_PDICT.values()))
BURSTSHAPE_PBOUNDS = np.array(list(DEFAULT_BURSTSHAPE_BOUNDS_PDICT.values()))


@jjit
def _get_burstshape_galpop_from_params(gal_logsm, gal_logssfr, burstshapepop_params):
    (
        burstshape_u_lgyr_peak_x0_logsm_x0,
        burstshape_u_lgyr_peak_q_logsm_x0,
        burstshape_u_lgyr_peak_ms_logsm_x0,
        burstshape_u_lgyr_peak_x0_logsm_ylo,
        burstshape_u_lgyr_peak_q_logsm_ylo,
        burstshape_u_lgyr_peak_ms_logsm_ylo,
        burstshape_u_lgyr_peak_x0_logsm_yhi,
        burstshape_u_lgyr_peak_q_logsm_yhi,
        burstshape_u_lgyr_peak_ms_logsm_yhi,
        burstshape_u_lgyr_max_logsm_x0,
        burstshape_u_lgyr_max_logsm_ylo,
        burstshape_u_lgyr_max_logsm_yhi,
    ) = burstshapepop_params

    u_lgyr_peak_x0 = _sigmoid(
        gal_logsm,
        burstshape_u_lgyr_peak_x0_logsm_x0,
        LGSM_K,
        burstshape_u_lgyr_peak_q_logsm_x0,
        burstshape_u_lgyr_peak_ms_logsm_x0,
    )
    u_lgyr_peak_ylo = _sigmoid(
        gal_logsm,
        burstshape_u_lgyr_peak_x0_logsm_ylo,
        LGSM_K,
        burstshape_u_lgyr_peak_q_logsm_ylo,
        burstshape_u_lgyr_peak_ms_logsm_ylo,
    )
    u_lgyr_peak_yhi = _sigmoid(
        gal_logsm,
        burstshape_u_lgyr_peak_x0_logsm_yhi,
        LGSM_K,
        burstshape_u_lgyr_peak_q_logsm_yhi,
        burstshape_u_lgyr_peak_ms_logsm_yhi,
    )

    u_lgyr_peak = _sigmoid(
        gal_logssfr, u_lgyr_peak_x0, LGSSFR_K, u_lgyr_peak_ylo, u_lgyr_peak_yhi
    )

    u_lgyr_max = _sigmoid(
        gal_logsm,
        burstshape_u_lgyr_max_logsm_x0,
        LGSM_K,
        burstshape_u_lgyr_max_logsm_ylo,
        burstshape_u_lgyr_max_logsm_yhi,
    )

    return u_lgyr_peak, u_lgyr_max


@jjit
def _get_burstshape_galpop_from_u_params(
    gal_logsm, gal_logssfr, burstshapepop_u_params
):
    burstshapepop_params = _get_bounded_burstshape_params(burstshapepop_u_params)
    u_lgyr_peak, u_lgyr_max = _get_burstshape_galpop_from_params(
        gal_logsm, gal_logssfr, burstshapepop_params
    )
    return u_lgyr_peak, u_lgyr_max


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - lax.log(lnarg) / k


@jjit
def _get_bounded_burstshape_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, 0.1, lo, hi)


@jjit
def _get_unbounded_burstshape_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, 0.1, lo, hi)


_C = (0, 0)
_get_burstshape_params_kern = jjit(vmap(_get_bounded_burstshape_param, in_axes=_C))
_get_burstshape_u_params_kern = jjit(vmap(_get_unbounded_burstshape_param, in_axes=_C))


@jjit
def _get_bounded_burstshape_params(u_params):
    params = _get_burstshape_params_kern(u_params, BURSTSHAPE_PBOUNDS)
    return params


@jjit
def _get_unbounded_burstshape_params(params):
    u_params = _get_burstshape_u_params_kern(params, BURSTSHAPE_PBOUNDS)
    return u_params


DEFAULT_BURSTSHAPE_U_PARAMS = np.array(
    _get_unbounded_burstshape_params(DEFAULT_BURSTSHAPE_PARAMS)
)
