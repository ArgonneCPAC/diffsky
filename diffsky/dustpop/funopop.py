"""
"""
from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

DEFAULT_FUNOPOP_PDICT = OrderedDict(
    funo_logfburst_x0_logsm_x0_young=10.0,
    funo_logfburst_x0_logsm_ylo_young=-3.0,
    funo_logfburst_x0_logsm_yhi_young=-2.0,
    funo_logfburst_ylo_logsm_x0_young=10.25,
    funo_logfburst_ylo_logsm_ylo_young=0.1,
    funo_logfburst_ylo_logsm_yhi_young=0.05,
    funo_logfburst_yhi_logsm_x0_young=10.75,
    funo_logfburst_yhi_logsm_ylo_young=0.3,
    funo_logfburst_yhi_logsm_yhi_young=0.15,
    funo_logssfr_x0_logsm_x0_old=10.25,
    funo_logssfr_x0_logsm_ylo_old=-10.5,
    funo_logssfr_x0_logsm_yhi_old=-11.25,
    funo_logssfr_ylo_logsm_x0_old=9.5,
    funo_logssfr_ylo_logsm_ylo_old=0.15,
    funo_logssfr_ylo_logsm_yhi_old=0.05,
    funo_logssfr_yhi_logsm_x0_old=10.75,
    funo_logssfr_yhi_logsm_ylo_old=0.1,
    funo_logssfr_yhi_logsm_yhi_old=0.25,
    funo_vs_ssp_age_x0_logsm_x0=10.0,
    funo_vs_ssp_age_x0_logsm_ylo=0.3,
    funo_vs_ssp_age_x0_logsm_yhi=0.05,
)

LOGSM_X0_BOUNDS = 9.0, 12.0
LOGSSFR_X0_BOUNDS = -12.5, -7.0
LOGFB_X0_BOUNDS = -5.0, -1.0
FUNO_BOUNDS = 0.0, 1.0

FUNOPOP_BOUNDS_PDICT = OrderedDict(
    funo_logfburst_x0_logsm_x0_young=LOGSM_X0_BOUNDS,
    funo_logfburst_x0_logsm_ylo_young=LOGFB_X0_BOUNDS,
    funo_logfburst_x0_logsm_yhi_young=LOGFB_X0_BOUNDS,
    funo_logfburst_ylo_logsm_x0_young=LOGSM_X0_BOUNDS,
    funo_logfburst_ylo_logsm_ylo_young=FUNO_BOUNDS,
    funo_logfburst_ylo_logsm_yhi_young=FUNO_BOUNDS,
    funo_logfburst_yhi_logsm_x0_young=LOGSM_X0_BOUNDS,
    funo_logfburst_yhi_logsm_ylo_young=FUNO_BOUNDS,
    funo_logfburst_yhi_logsm_yhi_young=FUNO_BOUNDS,
    funo_logssfr_x0_logsm_x0_old=LOGSM_X0_BOUNDS,
    funo_logssfr_x0_logsm_ylo_old=LOGSSFR_X0_BOUNDS,
    funo_logssfr_x0_logsm_yhi_old=LOGSSFR_X0_BOUNDS,
    funo_logssfr_ylo_logsm_x0_old=LOGSM_X0_BOUNDS,
    funo_logssfr_ylo_logsm_ylo_old=FUNO_BOUNDS,
    funo_logssfr_ylo_logsm_yhi_old=FUNO_BOUNDS,
    funo_logssfr_yhi_logsm_x0_old=LOGSM_X0_BOUNDS,
    funo_logssfr_yhi_logsm_ylo_old=FUNO_BOUNDS,
    funo_logssfr_yhi_logsm_yhi_old=FUNO_BOUNDS,
    funo_vs_ssp_age_x0_logsm_x0=LOGSM_X0_BOUNDS,
    funo_vs_ssp_age_x0_logsm_ylo=FUNO_BOUNDS,
    funo_vs_ssp_age_x0_logsm_yhi=FUNO_BOUNDS,
)

LGFRACUNO_LGSM_K = 5.0
LGFRACUNO_LGFB_K = 3.0
LGFRACUNO_LGSSFR_K = 5.0
LGFRACUNO_LGAGE_K = 5.0
BOUNDING_K = 0.1

LGFUNO_PLAW_SLOPE = 2.0

FunoPopParams = namedtuple("FunoPopParams", DEFAULT_FUNOPOP_PDICT.keys())
_FUNOPOP_UPNAMES = ["u_" + key for key in DEFAULT_FUNOPOP_PDICT.keys()]
FunoPopUParams = namedtuple("FunoPopUParams", _FUNOPOP_UPNAMES)

DEFAULT_FUNOPOP_PARAMS = FunoPopParams(**DEFAULT_FUNOPOP_PDICT)

FUNOPOP_PBOUNDS = FunoPopParams(**FUNOPOP_BOUNDS_PDICT)


@jjit
def _age_dep_funo_from_params_kern(
    gal_logsm,
    gal_logfburst,
    gal_logssfr,
    ssp_lg_age_gyr,
    funo_logfburst_x0_logsm_x0_young,
    funo_logfburst_x0_logsm_ylo_young,
    funo_logfburst_x0_logsm_yhi_young,
    funo_logfburst_ylo_logsm_x0_young,
    funo_logfburst_ylo_logsm_ylo_young,
    funo_logfburst_ylo_logsm_yhi_young,
    funo_logfburst_yhi_logsm_x0_young,
    funo_logfburst_yhi_logsm_ylo_young,
    funo_logfburst_yhi_logsm_yhi_young,
    funo_logssfr_x0_logsm_x0_old,
    funo_logssfr_x0_logsm_ylo_old,
    funo_logssfr_x0_logsm_yhi_old,
    funo_logssfr_ylo_logsm_x0_old,
    funo_logssfr_ylo_logsm_ylo_old,
    funo_logssfr_ylo_logsm_yhi_old,
    funo_logssfr_yhi_logsm_x0_old,
    funo_logssfr_yhi_logsm_ylo_old,
    funo_logssfr_yhi_logsm_yhi_old,
    funo_vs_ssp_age_x0_logsm_x0,
    funo_vs_ssp_age_x0_logsm_ylo,
    funo_vs_ssp_age_x0_logsm_yhi,
):
    #  compute params for young stars

    funo_logsm_x0_young = _sigmoid(
        gal_logsm,
        funo_logfburst_x0_logsm_x0_young,
        LGFRACUNO_LGSM_K,
        funo_logfburst_x0_logsm_ylo_young,
        funo_logfburst_x0_logsm_yhi_young,
    )

    funo_logsm_ylo_young = _sigmoid(
        gal_logsm,
        funo_logfburst_ylo_logsm_x0_young,
        LGFRACUNO_LGSM_K,
        funo_logfburst_ylo_logsm_ylo_young,
        funo_logfburst_ylo_logsm_yhi_young,
    )

    funo_logsm_yhi_young = _sigmoid(
        gal_logsm,
        funo_logfburst_yhi_logsm_x0_young,
        LGFRACUNO_LGSM_K,
        funo_logfburst_yhi_logsm_ylo_young,
        funo_logfburst_yhi_logsm_yhi_young,
    )

    funo_vs_ssp_age_young = _sigmoid(
        gal_logfburst,
        funo_logsm_x0_young,
        LGFRACUNO_LGFB_K,
        funo_logsm_ylo_young,
        funo_logsm_yhi_young,
    )

    #  compute params for old stars
    funo_logsm_x0_old = _sigmoid(
        gal_logsm,
        funo_logssfr_x0_logsm_x0_old,
        LGFRACUNO_LGSM_K,
        funo_logssfr_x0_logsm_ylo_old,
        funo_logssfr_x0_logsm_yhi_old,
    )

    funo_logsm_ylo_old = _sigmoid(
        gal_logsm,
        funo_logssfr_ylo_logsm_x0_old,
        LGFRACUNO_LGSM_K,
        funo_logssfr_ylo_logsm_ylo_old,
        funo_logssfr_ylo_logsm_yhi_old,
    )

    funo_logsm_yhi_old = _sigmoid(
        gal_logsm,
        funo_logssfr_yhi_logsm_x0_old,
        LGFRACUNO_LGSM_K,
        funo_logssfr_yhi_logsm_ylo_old,
        funo_logssfr_yhi_logsm_yhi_old,
    )

    funo_vs_ssp_age_old = _sigmoid(
        gal_logssfr,
        funo_logsm_x0_old,
        LGFRACUNO_LGSSFR_K,
        funo_logsm_ylo_old,
        funo_logsm_yhi_old,
    )

    #  compute the lg_age transition between young and old stars

    funo_vs_ssp_age_x0 = _sigmoid(
        gal_logsm,
        funo_vs_ssp_age_x0_logsm_x0,
        LGFRACUNO_LGSM_K,
        funo_vs_ssp_age_x0_logsm_ylo,
        funo_vs_ssp_age_x0_logsm_yhi,
    )

    #  compute parameter b, the high-end cutoff in the power-law PDF P(F_uno)

    funo = _sigmoid(
        ssp_lg_age_gyr,
        funo_vs_ssp_age_x0,
        LGFRACUNO_LGAGE_K,
        funo_vs_ssp_age_young,
        funo_vs_ssp_age_old,
    )

    return funo


@jjit
def calc_age_dep_funo_from_params_scalar(
    funopop_params, gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr
):
    return _age_dep_funo_from_params_kern(
        gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, *funopop_params
    )


N_PARAMS = len(DEFAULT_FUNOPOP_PARAMS)
_a = [0, 0, 0, None, *[None] * N_PARAMS]
_b = [None, None, None, 0, *[None] * N_PARAMS]
_get_funo_from_funopop_params_kern = jjit(
    vmap(_age_dep_funo_from_params_kern, in_axes=_b)
)
_get_funo_from_funopop_params_galpop_kern = jjit(
    vmap(_get_funo_from_funopop_params_kern, in_axes=_a)
)


@jjit
def get_funo_from_funopop_params(
    funopop_params, gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr
):
    funopop_params = jnp.array(
        [getattr(funopop_params, pname) for pname in DEFAULT_FUNOPOP_PARAMS._fields]
    )
    return _get_funo_from_funopop_params_kern(
        gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, *funopop_params
    )


@jjit
def get_funo_from_funopop_u_params(
    funopop_u_params, gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr
):
    funopop_params = get_bounded_funopop_params(funopop_u_params)
    return get_funo_from_funopop_params(
        funopop_params, gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr
    )


@jjit
def get_funo_from_funopop_params_galpop(
    funopop_params, gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr
):
    return _get_funo_from_funopop_params_galpop_kern(
        gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr, *funopop_params
    )


@jjit
def get_funo_from_funopop_u_params_galpop(
    funo_u_params, gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr
):
    funopop_params = get_bounded_funopop_params(funo_u_params)
    return get_funo_from_funopop_params_galpop(
        funopop_params, gal_logsm, gal_logfburst, gal_logssfr, ssp_lg_age_gyr
    )


@jjit
def _get_p_from_u_p_scalar(u_p, bounds):
    lo, hi = bounds
    p0 = 0.5 * (lo + hi)
    p = _sigmoid(u_p, p0, BOUNDING_K, lo, hi)
    return p


@jjit
def _get_u_p_from_p_scalar(p, bounds):
    lo, hi = bounds
    p0 = 0.5 * (lo + hi)
    u_p = _inverse_sigmoid(p, p0, BOUNDING_K, lo, hi)
    return u_p


_get_p_from_u_p_vmap = jjit(vmap(_get_p_from_u_p_scalar, in_axes=(0, 0)))
_get_u_p_from_p_vmap = jjit(vmap(_get_u_p_from_p_scalar, in_axes=(0, 0)))


@jjit
def get_bounded_funopop_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _FUNOPOP_UPNAMES])
    params = _get_p_from_u_p_vmap(jnp.array(u_params), jnp.array(FUNOPOP_PBOUNDS))
    return FunoPopParams(*params)


@jjit
def get_unbounded_funopop_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_FUNOPOP_PARAMS._fields]
    )
    u_params = _get_u_p_from_p_vmap(jnp.array(params), jnp.array(FUNOPOP_PBOUNDS))
    return FunoPopUParams(*u_params)


DEFAULT_FUNOPOP_U_PARAMS = FunoPopUParams(
    *get_unbounded_funopop_params(DEFAULT_FUNOPOP_PARAMS)
)
