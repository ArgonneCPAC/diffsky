"""
"""
from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

LGSM_K = 5.0
LGSSFR_K = 5.0

DEFAULT_AVPOP_PDICT = OrderedDict(
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

AVPOP_PBOUNDS_PDICT = OrderedDict(
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

AvPopParams = namedtuple("AvPopParams", DEFAULT_AVPOP_PDICT.keys())

_AVPOP_UPNAMES = ["u_" + key for key in AVPOP_PBOUNDS_PDICT.keys()]
AvPopUParams = namedtuple("AvPopUParams", _AVPOP_UPNAMES)


DEFAULT_AVPOP_PARAMS = AvPopParams(**DEFAULT_AVPOP_PDICT)
AVPOP_PBOUNDS = AvPopParams(**AVPOP_PBOUNDS_PDICT)


@jjit
def get_av_from_avpop_u_params(avpop_u_params, logsm, logssfr):
    avpop_params = get_bounded_avpop_params(avpop_u_params)
    av = get_av_from_avpop_params(avpop_params, logsm, logssfr)
    return av


@jjit
def get_av_from_avpop_params(avpop_params, logsm, logssfr):
    lgav_logssfr_x0 = _sigmoid(
        logsm,
        avpop_params.lgav_logsm_x0_x0,
        LGSM_K,
        avpop_params.lgav_logsm_ylo_x0,
        avpop_params.lgav_logsm_yhi_x0,
    )
    lgav_logssfr_q = _sigmoid(
        logsm,
        avpop_params.lgav_logsm_x0_q,
        LGSM_K,
        avpop_params.lgav_logsm_ylo_q,
        avpop_params.lgav_logsm_yhi_q,
    )
    lgav_logssfr_ms = _sigmoid(
        logsm,
        avpop_params.lgav_logsm_x0_ms,
        LGSSFR_K,
        avpop_params.lgav_logsm_ylo_ms,
        avpop_params.lgav_logsm_yhi_ms,
    )

    lgav = _sigmoid(
        logssfr,
        lgav_logssfr_x0,
        LGSSFR_K,
        lgav_logssfr_q,
        lgav_logssfr_ms,
    )
    return 10**lgav


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
_get_avpop_params_kern = jjit(vmap(_get_bounded_lgav_param, in_axes=_C))
_get_lgav_u_params_kern = jjit(vmap(_get_unbounded_lgav_param, in_axes=_C))


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
    u_params = _get_lgav_u_params_kern(jnp.array(params), jnp.array(AVPOP_PBOUNDS))
    return AvPopUParams(*u_params)


DEFAULT_AVPOP_U_PARAMS = AvPopUParams(*get_unbounded_avpop_params(DEFAULT_AVPOP_PARAMS))
