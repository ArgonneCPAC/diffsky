"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
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
    lgav_logsm_x0_x0=10.0,
    lgav_logsm_x0_q=10.5,
    lgav_logsm_x0_ms=9.5,
    lgav_logsm_ylo_x0=-10.25,
    lgav_logsm_ylo_q_z_ylo=-0.25,
    lgav_logsm_ylo_ms_z_ylo=-0.25,
    lgav_logsm_ylo_q_z_yhi=-0.25,
    lgav_logsm_ylo_ms_z_yhi=-0.25,
    lgav_logsm_yhi_x0=-11.25,
    lgav_logsm_yhi_q_z_ylo=-0.25,
    lgav_logsm_yhi_ms_z_ylo=-0.25,
    lgav_logsm_yhi_q_z_yhi=-0.25,
    lgav_logsm_yhi_ms_z_yhi=-0.25,
    lgav_z_x0=1.0,
    av_boost_logsm_lo_q=0.25,
    av_boost_logsm_hi_q=0.25,
    av_boost_logsm_lo_ms=0.25,
    av_boost_logsm_hi_ms=0.25,
)

LGSM_X0_BOUNDS = (8.0, 12.0)
LGSSFR_X0_BOUNDS = (-13.0, -7.0)
LGAV_BOUNDS = (-4.0, 1.5)
REDSHIFT_BOUNDS = (0.0, 5.0)
DELTA_AV_BOUNDS = (0.0, 3.0)

AVPOP_PBOUNDS_PDICT = OrderedDict(
    lgav_logsm_x0_x0=LGSM_X0_BOUNDS,
    lgav_logsm_x0_q=LGSM_X0_BOUNDS,
    lgav_logsm_x0_ms=LGSM_X0_BOUNDS,
    lgav_logsm_ylo_x0=LGSSFR_X0_BOUNDS,
    lgav_logsm_ylo_q_z_ylo=LGAV_BOUNDS,
    lgav_logsm_ylo_ms_z_ylo=LGAV_BOUNDS,
    lgav_logsm_ylo_q_z_yhi=LGAV_BOUNDS,
    lgav_logsm_ylo_ms_z_yhi=LGAV_BOUNDS,
    lgav_logsm_yhi_x0=LGSSFR_X0_BOUNDS,
    lgav_logsm_yhi_q_z_ylo=LGAV_BOUNDS,
    lgav_logsm_yhi_ms_z_ylo=LGAV_BOUNDS,
    lgav_logsm_yhi_q_z_yhi=LGAV_BOUNDS,
    lgav_logsm_yhi_ms_z_yhi=LGAV_BOUNDS,
    lgav_z_x0=REDSHIFT_BOUNDS,
    av_boost_logsm_lo_q=DELTA_AV_BOUNDS,
    av_boost_logsm_hi_q=DELTA_AV_BOUNDS,
    av_boost_logsm_lo_ms=DELTA_AV_BOUNDS,
    av_boost_logsm_hi_ms=DELTA_AV_BOUNDS,
)


AvPopParams = namedtuple("AvPopParams", DEFAULT_AVPOP_PDICT.keys())

_AVPOP_UPNAMES = ["u_" + key for key in AVPOP_PBOUNDS_PDICT.keys()]
AvPopUParams = namedtuple("AvPopUParams", _AVPOP_UPNAMES)


DEFAULT_AVPOP_PARAMS = AvPopParams(**DEFAULT_AVPOP_PDICT)
AVPOP_PBOUNDS = AvPopParams(**AVPOP_PBOUNDS_PDICT)


@jjit
def get_av_from_avpop_params_scalar(avpop_params, logsm, logssfr, redshift, lg_age_gyr):
    lgav_logssfr_x0 = _sigmoid(
        logsm,
        avpop_params.lgav_logsm_x0_x0,
        LGSM_K,
        avpop_params.lgav_logsm_ylo_x0,
        avpop_params.lgav_logsm_yhi_x0,
    )
    lgav_logssfr_q_z_ylo = _sigmoid(
        logsm,
        avpop_params.lgav_logsm_x0_q,
        LGSM_K,
        avpop_params.lgav_logsm_ylo_q_z_ylo,
        avpop_params.lgav_logsm_yhi_q_z_ylo,
    )
    lgav_logssfr_ms_z_ylo = _sigmoid(
        logsm,
        avpop_params.lgav_logsm_x0_ms,
        LGSSFR_K,
        avpop_params.lgav_logsm_ylo_ms_z_ylo,
        avpop_params.lgav_logsm_yhi_ms_z_ylo,
    )

    lgav_z_ylo = _sigmoid(
        logssfr,
        lgav_logssfr_x0,
        LGSSFR_K,
        lgav_logssfr_q_z_ylo,
        lgav_logssfr_ms_z_ylo,
    )

    lgav_logssfr_q_z_yhi = _sigmoid(
        logsm,
        avpop_params.lgav_logsm_x0_q,
        LGSM_K,
        avpop_params.lgav_logsm_ylo_q_z_yhi,
        avpop_params.lgav_logsm_yhi_q_z_yhi,
    )
    lgav_logssfr_ms_z_yhi = _sigmoid(
        logsm,
        avpop_params.lgav_logsm_x0_ms,
        LGSSFR_K,
        avpop_params.lgav_logsm_ylo_ms_z_yhi,
        avpop_params.lgav_logsm_yhi_ms_z_yhi,
    )

    lgav_z_yhi = _sigmoid(
        logssfr,
        lgav_logssfr_x0,
        LGSSFR_K,
        lgav_logssfr_q_z_yhi,
        lgav_logssfr_ms_z_yhi,
    )

    lgav = _sigmoid(
        redshift,
        avpop_params.lgav_z_x0,
        REDSHIFT_K,
        lgav_z_ylo,
        lgav_z_yhi,
    )
    av = 10**lgav

    av_boost = _get_young_star_av_boost(
        logsm,
        logssfr,
        avpop_params.av_boost_logsm_lo_q,
        avpop_params.av_boost_logsm_hi_q,
        avpop_params.av_boost_logsm_lo_ms,
        avpop_params.av_boost_logsm_hi_ms,
    )
    delta_av = _young_star_av_boost_kern(lg_age_gyr, av_boost)

    return av + delta_av


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


@jjit
def _young_star_av_boost_kern(lg_age_gyr, av_boost):
    return _sigmoid(lg_age_gyr, LGAGE_GYR_X0, LGAGE_K, av_boost, 0)


@jjit
def _get_young_star_av_boost(
    logsm,
    logssfr,
    av_boost_logsm_lo_q,
    av_boost_logsm_hi_q,
    av_boost_logsm_lo_ms,
    av_boost_logsm_hi_ms,
):
    av_boost_q = _sigmoid(
        logsm, LGSM_X0, LGSM_K, av_boost_logsm_lo_q, av_boost_logsm_hi_q
    )
    av_boost_ms = _sigmoid(
        logsm, LGSM_X0, LGSM_K, av_boost_logsm_lo_ms, av_boost_logsm_hi_ms
    )
    av_boost = _sigmoid(logssfr, LGSSFR_X0, LGSSFR_K, av_boost_q, av_boost_ms)
    return av_boost


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
