"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import nn
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

REDSHIFT_K = 5.0
LGSM_K = 5.0
LGSSFR_K = 5.0
LGAGE_K = 5.0

LGAGE_GYR_X0 = 6.5 - 9.0
LGSM_X0 = 10.0
LGSSFR_X0 = -10.5

DEFAULT_AVPOP_PDICT = OrderedDict(
    lgav_logsm_x0=10.0,
    lgav_logssfr_x0=-10.25,
    lgav_logsm_ylo_q_z_ylo=-0.25,
    lgav_logsm_ylo_ms_z_ylo=-0.25,
    lgav_logsm_ylo_q_z_yhi=-0.25,
    lgav_logsm_ylo_ms_z_yhi=-0.25,
    lgav_logsm_yhi_q_z_ylo=-0.25,
    lgav_logsm_yhi_ms_z_ylo=-0.25,
    lgav_logsm_yhi_q_z_yhi=-0.25,
    lgav_logsm_yhi_ms_z_yhi=-0.25,
    lgav_z_x0=1.0,
    delta_u_av=0.2,
)

LGSM_X0_BOUNDS = (8.0, 12.0)
LGSSFR_X0_BOUNDS = (-13.0, -7.0)
LGAV_BOUNDS = (-4.0, 1.5)
REDSHIFT_BOUNDS = (0.0, 5.0)
DELTA_U_AV_BOUNDS = (0.0, 1.0)

AVPOP_PBOUNDS_PDICT = OrderedDict(
    lgav_logsm_x0=LGSM_X0_BOUNDS,
    lgav_logssfr_x0=LGSSFR_X0_BOUNDS,
    lgav_logsm_ylo_q_z_ylo=LGAV_BOUNDS,
    lgav_logsm_ylo_ms_z_ylo=LGAV_BOUNDS,
    lgav_logsm_ylo_q_z_yhi=LGAV_BOUNDS,
    lgav_logsm_ylo_ms_z_yhi=LGAV_BOUNDS,
    lgav_logsm_yhi_q_z_ylo=LGAV_BOUNDS,
    lgav_logsm_yhi_ms_z_ylo=LGAV_BOUNDS,
    lgav_logsm_yhi_q_z_yhi=LGAV_BOUNDS,
    lgav_logsm_yhi_ms_z_yhi=LGAV_BOUNDS,
    lgav_z_x0=REDSHIFT_BOUNDS,
    delta_u_av=DELTA_U_AV_BOUNDS,
)


AvPopParams = namedtuple("AvPopParams", DEFAULT_AVPOP_PDICT.keys())

_AVPOP_UPNAMES = ["u_" + key for key in AVPOP_PBOUNDS_PDICT.keys()]
AvPopUParams = namedtuple("AvPopUParams", _AVPOP_UPNAMES)


DEFAULT_AVPOP_PARAMS = AvPopParams(**DEFAULT_AVPOP_PDICT)
AVPOP_PBOUNDS = AvPopParams(**AVPOP_PBOUNDS_PDICT)


@jjit
def get_av_from_avpop_params_scalar(avpop_params, logsm, logssfr, redshift, lg_age_gyr):
    lgav_logssfr_q_z_ylo = _sigmoid(
        logsm,
        avpop_params.lgav_logsm_x0,
        LGSM_K,
        avpop_params.lgav_logsm_ylo_q_z_ylo,
        avpop_params.lgav_logsm_yhi_q_z_ylo,
    )
    lgav_logssfr_ms_z_ylo = _sigmoid(
        logsm,
        avpop_params.lgav_logsm_x0,
        LGSSFR_K,
        avpop_params.lgav_logsm_ylo_ms_z_ylo,
        avpop_params.lgav_logsm_yhi_ms_z_ylo,
    )

    lgav_z_ylo = _sigmoid(
        logssfr,
        avpop_params.lgav_logssfr_x0,
        LGSSFR_K,
        lgav_logssfr_q_z_ylo,
        lgav_logssfr_ms_z_ylo,
    )

    lgav_logssfr_q_z_yhi = _sigmoid(
        logsm,
        avpop_params.lgav_logsm_x0,
        LGSM_K,
        avpop_params.lgav_logsm_ylo_q_z_yhi,
        avpop_params.lgav_logsm_yhi_q_z_yhi,
    )
    lgav_logssfr_ms_z_yhi = _sigmoid(
        logsm,
        avpop_params.lgav_logsm_x0,
        LGSSFR_K,
        avpop_params.lgav_logsm_ylo_ms_z_yhi,
        avpop_params.lgav_logsm_yhi_ms_z_yhi,
    )

    lgav_z_yhi = _sigmoid(
        logssfr,
        avpop_params.lgav_logssfr_x0,
        LGSSFR_K,
        lgav_logssfr_q_z_yhi,
        lgav_logssfr_ms_z_yhi,
    )

    u_av = _sigmoid(
        redshift, avpop_params.lgav_z_x0, REDSHIFT_K, lgav_z_ylo, lgav_z_yhi
    )

    delta_u_av = _young_star_av_boost_kern(lg_age_gyr, avpop_params.u_av_boost_ostars)
    u_av = u_av + delta_u_av

    av = nn.softplus(u_av)

    return av


@jjit
def _young_star_av_boost_kern(lg_age_gyr, u_av_boost_ostars):
    return _sigmoid(lg_age_gyr, LGAGE_GYR_X0, LGAGE_K, u_av_boost_ostars, 0)


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
