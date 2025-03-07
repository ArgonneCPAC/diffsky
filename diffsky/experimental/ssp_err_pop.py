"""
"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..systematics import ssp_errors
from ..utils import _inverse_sigmoid, _sigmoid

LGSSFR_K = 5.0
LGSSFR_X0_BOUNDS = (-11.5, -8.0)
BOUNDING_K = 0.1

DEFAULT_SSP_ERR_POP_PDICT = OrderedDict()
DEFAULT_SSP_ERR_POP_PDICT["ssp_ff_x0"] = 3.7
DEFAULT_SSP_ERR_POP_PDICT["ssp_ff_lgssfr_x0"] = -10.0
DEFAULT_SSP_ERR_POP_PDICT["ssp_ff_ylo_ylo"] = 0.0
DEFAULT_SSP_ERR_POP_PDICT["ssp_ff_ylo_yhi"] = 0.0
DEFAULT_SSP_ERR_POP_PDICT["ssp_ff_yhi_ylo"] = 0.0
DEFAULT_SSP_ERR_POP_PDICT["ssp_ff_yhi_yhi"] = 0.0

SSPerrPopParams = namedtuple("SSPErrPopParams", list(DEFAULT_SSP_ERR_POP_PDICT.keys()))
DEFAULT_SSP_ERR_POP_PARAMS = SSPerrPopParams(**DEFAULT_SSP_ERR_POP_PDICT)

U_PNAMES = ["u_" + key for key in DEFAULT_SSP_ERR_POP_PARAMS._fields]
SSPerrPopUParams = namedtuple("SSPerrPopUParams", U_PNAMES)

SSP_ERR_POP_BOUNDS_DICT = OrderedDict(
    ssp_ff_x0=ssp_errors.SSPERR_PBOUNDS.ssp_ff_x0,
    ssp_ff_lgssfr_x0=LGSSFR_X0_BOUNDS,
    ssp_ff_ylo_ylo=ssp_errors.SSPERR_PBOUNDS.ssp_ff_ylo,
    ssp_ff_ylo_yhi=ssp_errors.SSPERR_PBOUNDS.ssp_ff_ylo,
    ssp_ff_yhi_ylo=ssp_errors.SSPERR_PBOUNDS.ssp_ff_yhi,
    ssp_ff_yhi_yhi=ssp_errors.SSPERR_PBOUNDS.ssp_ff_yhi,
)
SSPERR_POP_PBOUNDS = SSPerrPopParams(**SSP_ERR_POP_BOUNDS_DICT)


@jjit
def get_flux_factor_from_lgssfr_kern(ssp_err_pop_params, lgssfr, wave):
    ssp_err_params = get_ssp_err_params_from_lgssfr_kern(ssp_err_pop_params, lgssfr)
    flux_factor = ssp_errors.ssp_flux_factor(ssp_err_params, wave)
    return flux_factor


@jjit
def get_ssp_err_params_from_lgssfr_kern(ssp_err_pop_params, lgssfr):

    ssp_ff_x0 = ssp_err_pop_params.ssp_ff_x0
    ssp_ff_ylo = _get_ssp_err_ff_ylo(ssp_err_pop_params, lgssfr)
    ssp_ff_yhi = _get_ssp_err_ff_yhi(ssp_err_pop_params, lgssfr)
    ssp_err_params = ssp_errors.DEFAULT_SSPERR_PARAMS._make(
        (ssp_ff_x0, ssp_ff_ylo, ssp_ff_yhi)
    )
    return ssp_err_params


@jjit
def _get_ssp_err_ff_ylo(ssp_err_pop_params, lgssfr):
    ssp_ff_ylo = _sigmoid(
        lgssfr,
        ssp_err_pop_params.ssp_ff_lgssfr_x0,
        LGSSFR_K,
        ssp_err_pop_params.ssp_ff_ylo_ylo,
        ssp_err_pop_params.ssp_ff_ylo_yhi,
    )
    return ssp_ff_ylo


@jjit
def _get_ssp_err_ff_yhi(ssp_err_pop_params, lgssfr):
    ssp_ff_yhi = _sigmoid(
        lgssfr,
        ssp_err_pop_params.ssp_ff_lgssfr_x0,
        LGSSFR_K,
        ssp_err_pop_params.ssp_ff_yhi_ylo,
        ssp_err_pop_params.ssp_ff_yhi_yhi,
    )
    return ssp_ff_yhi


@jjit
def _get_bounded_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, BOUNDING_K, lo, hi)


@jjit
def _get_unbounded_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, BOUNDING_K, lo, hi)


_C = (0, 0)
_get_params_kern = jjit(vmap(_get_bounded_param, in_axes=_C))
_get_u_params_kern = jjit(vmap(_get_unbounded_param, in_axes=_C))


@jjit
def get_bounded_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in U_PNAMES])
    params = _get_params_kern(jnp.array(u_params), jnp.array(SSPERR_POP_PBOUNDS))
    return SSPerrPopParams(*params)


@jjit
def get_unbounded_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_SSP_ERR_POP_PARAMS._fields]
    )
    u_params = _get_u_params_kern(jnp.array(params), jnp.array(SSPERR_POP_PBOUNDS))
    return SSPerrPopUParams(*u_params)


DEFAULT_SSP_ERR_POP_U_PARAMS = SSPerrPopUParams(
    *get_unbounded_params(DEFAULT_SSP_ERR_POP_PARAMS)
)
