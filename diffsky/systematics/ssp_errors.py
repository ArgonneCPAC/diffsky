"""Model for the fractional error in the SSP SEDs in optical wavelengths.

The primary kernel is ssp_flux_factor, which supplies a multiplicative factor
that can be applied to any SSP flux table:

    new_ssp_flux = flux_factor*orig_ssp_flux

Under the hood, the ssp_flux_factor is parameterized by a sigmoid in log-log space:

    * ssp_ff_x0 sets log10(wave) of the sigmoid inflection point
    * ssp_ff_ylo sets log10(flux_factor) at small wavelengths
    * ssp_ff_yhi sets log10(flux_factor) at large wavelengths

"""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid

DEFAULT_SSPERR_PDICT = OrderedDict(ssp_ff_x0=3.5, ssp_ff_ylo=0.0, ssp_ff_yhi=0.0)
SSPerrParams = namedtuple("SSPerrParams", DEFAULT_SSPERR_PDICT.keys())
DEFAULT_SSPERR_PARAMS = SSPerrParams(**DEFAULT_SSPERR_PDICT)
U_PNAMES = ["u_" + key for key in DEFAULT_SSPERR_PARAMS._fields]
SSPerrUParams = namedtuple("SSPerrUParams", U_PNAMES)

SSPERR_FF_BOUNDS = (-0.75, 0.75)
SSPERR_BOUNDS_DICT = OrderedDict(
    ssp_ff_x0=(3.1, 4.1), ssp_ff_ylo=SSPERR_FF_BOUNDS, ssp_ff_yhi=SSPERR_FF_BOUNDS
)
SSPERR_PBOUNDS = SSPerrParams(**SSPERR_BOUNDS_DICT)

BOUNDING_K = 0.1
LGAA_K = 15.0


@jjit
def ssp_flux_factor(params, wave):
    """Fractional change to the SSP SEDs as a function of wavelength

    Parameters
    ----------
    params : namedtuple
        See DEFAULT_SSPERR_PARAMS for an example

    wave : array, shape (n_wave, )
        Array of λ in angstrom

    Returns
    -------
    flux_factor : array, shape (n_wave, )
        new_ssp_flux = flux_factor*orig_ssp_flux

    """
    flux_factor = 1.0 + _ssp_flux_factor_kern(params, jnp.log10(wave))
    return flux_factor


@jjit
def _ssp_flux_factor_kern(params, lgwave):
    flux_factor = _sigmoid(
        lgwave, params.ssp_ff_x0, LGAA_K, params.ssp_ff_ylo, params.ssp_ff_yhi
    )
    return flux_factor


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
    params = _get_params_kern(jnp.array(u_params), jnp.array(SSPERR_PBOUNDS))
    return SSPerrParams(*params)


@jjit
def get_unbounded_params(params):
    params = jnp.array(
        [getattr(params, pname) for pname in DEFAULT_SSPERR_PARAMS._fields]
    )
    u_params = _get_u_params_kern(jnp.array(params), jnp.array(SSPERR_PBOUNDS))
    return SSPerrUParams(*u_params)


DEFAULT_SSPERR_U_PARAMS = SSPerrUParams(*get_unbounded_params(DEFAULT_SSPERR_PARAMS))
