"""Module implements smhm_kernel: a three-roll stellar-to-halo-mass relation.
This module is just a namedtuple-based wrapper around threeroll_kernels.py.
"""

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from . import threeroll_kernels as std

Params = namedtuple("Params", std.DEFAULT_SMHM_PARAMS.keys())
DEFAULT_PARAMS = Params(**std.DEFAULT_SMHM_PARAMS)

PBOUNDS = Params(**std.DEFAULT_SMHM_PARAM_BOUNDS)
UParams = namedtuple("UParams", ["u_" + key for key in Params._fields])


@jjit
def smhm_kernel(params, logmh):
    """Three-roll tellar-to-halo mass relation

    Parameters
    ----------
    params : namedtuple

    logmh : array, shape (n, )

    Returns
    -------
    logsm : array, shape (n, )

    """
    params = jnp.array([getattr(params, pname) for pname in Params._fields])
    return std._logsm_from_logmh(params, logmh)


@jjit
def smhm_kernel_u_params(u_params, logmh):
    """ """
    params = get_bounded_params(u_params)
    return std._logsm_from_logmh(params, logmh)


@jjit
def _get_bounded_param(u_param, bound):
    lo, hi = bound
    return std._sigmoid(u_param, std._PBOUND_X0, std._PBOUND_K, lo, hi)


@jjit
def _get_unbounded_param(param, bound):
    lo, hi = bound
    return std._inverse_sigmoid(param, std._PBOUND_X0, std._PBOUND_K, lo, hi)


_C = (0, 0)
_get_params_kern = jjit(vmap(_get_bounded_param, in_axes=_C))
_get_u_params_kern = jjit(vmap(_get_unbounded_param, in_axes=_C))


@jjit
def get_bounded_params(u_params):
    u_params = jnp.array([getattr(u_params, u_pname) for u_pname in UParams._fields])
    params = _get_params_kern(jnp.array(u_params), jnp.array(std.BOUNDS))
    return Params(*params)


@jjit
def get_unbounded_params(params):
    params = jnp.array([getattr(params, pname) for pname in Params._fields])
    u_params = _get_u_params_kern(jnp.array(params), jnp.array(std.BOUNDS))
    return UParams(*u_params)


DEFAULT_U_PARAMS = UParams(*get_unbounded_params(DEFAULT_PARAMS))
