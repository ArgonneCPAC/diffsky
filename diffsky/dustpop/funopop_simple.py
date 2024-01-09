"""
"""
from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

DEFAULT_FUNOPOP_PDICT = OrderedDict(
    funo_logssfr_x0_logsm_x0=10.0,
    funo_logssfr_x0_logsm_ylo=-11.0,
    funo_logssfr_x0_logsm_yhi=-11.0,
    funo_logssfr_ylo_logsm_x0=10.25,
    funo_logssfr_ylo_logsm_ylo=0.1,
    funo_logssfr_ylo_logsm_yhi=0.05,
    funo_logssfr_yhi_logsm_x0=10.75,
    funo_logssfr_yhi_logsm_ylo=0.3,
    funo_logssfr_yhi_logsm_yhi=0.15,
)

LOGSM_X0_BOUNDS = 9.0, 12.0
LOGSSFR_X0_BOUNDS = -12.5, -7.0
FUNO_BOUNDS = 0.0, 0.5

FUNOPOP_BOUNDS_PDICT = OrderedDict(
    funo_logssfr_x0_logsm_x0=LOGSM_X0_BOUNDS,
    funo_logssfr_x0_logsm_ylo=LOGSSFR_X0_BOUNDS,
    funo_logssfr_x0_logsm_yhi=LOGSSFR_X0_BOUNDS,
    funo_logssfr_ylo_logsm_x0=LOGSM_X0_BOUNDS,
    funo_logssfr_ylo_logsm_ylo=FUNO_BOUNDS,
    funo_logssfr_ylo_logsm_yhi=FUNO_BOUNDS,
    funo_logssfr_yhi_logsm_x0=LOGSM_X0_BOUNDS,
    funo_logssfr_yhi_logsm_ylo=FUNO_BOUNDS,
    funo_logssfr_yhi_logsm_yhi=FUNO_BOUNDS,
)

LGSM_K = 5.0
LGSSFR_K = 5.0
BOUNDING_K = 0.1

FunoPopParams = namedtuple("FunoPopParams", DEFAULT_FUNOPOP_PDICT.keys())
_FUNOPOP_UPNAMES = ["u_" + key for key in DEFAULT_FUNOPOP_PDICT.keys()]
FunoPopUParams = namedtuple("FunoPopUParams", _FUNOPOP_UPNAMES)

DEFAULT_FUNOPOP_PARAMS = FunoPopParams(**DEFAULT_FUNOPOP_PDICT)

FUNOPOP_PBOUNDS = FunoPopParams(**FUNOPOP_BOUNDS_PDICT)


@jjit
def _funo_from_params_kern(
    logsm,
    logssfr,
    funo_logssfr_x0_logsm_x0,
    funo_logssfr_x0_logsm_ylo,
    funo_logssfr_x0_logsm_yhi,
    funo_logssfr_ylo_logsm_x0,
    funo_logssfr_ylo_logsm_ylo,
    funo_logssfr_ylo_logsm_yhi,
    funo_logssfr_yhi_logsm_x0,
    funo_logssfr_yhi_logsm_ylo,
    funo_logssfr_yhi_logsm_yhi,
):
    funo_logssfr_x0 = _sigmoid(
        logsm,
        funo_logssfr_x0_logsm_x0,
        LGSM_K,
        funo_logssfr_x0_logsm_ylo,
        funo_logssfr_x0_logsm_yhi,
    )

    funo_logsm_q = _sigmoid(
        logsm,
        funo_logssfr_ylo_logsm_x0,
        LGSM_K,
        funo_logssfr_ylo_logsm_ylo,
        funo_logssfr_ylo_logsm_yhi,
    )

    funo_logsm_ms = _sigmoid(
        logsm,
        funo_logssfr_yhi_logsm_x0,
        LGSM_K,
        funo_logssfr_yhi_logsm_ylo,
        funo_logssfr_yhi_logsm_yhi,
    )

    funo = _sigmoid(logssfr, funo_logssfr_x0, LGSSFR_K, funo_logsm_q, funo_logsm_ms)

    return funo


@jjit
def get_funo_from_funopop_params(funopop_params, gal_logsm, gal_logssfr):
    funopop_params = jnp.array(
        [getattr(funopop_params, pname) for pname in DEFAULT_FUNOPOP_PARAMS._fields]
    )
    return _funo_from_params_kern(gal_logsm, gal_logssfr, *funopop_params)


@jjit
def get_funo_from_funopop_u_params(funopop_u_params, gal_logsm, gal_logssfr):
    funopop_params = get_bounded_funopop_params(funopop_u_params)
    return get_funo_from_funopop_params(funopop_params, gal_logsm, gal_logssfr)


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
