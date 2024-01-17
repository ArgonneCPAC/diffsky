"""The lg_hmf_kern function gives a differentiable prediction for
the cumulative Host Halo Mass Function (HMF), <Nhalos(>mp)>,
here mp is the peak historical mass of the main progenitor halo.

"""
from collections import OrderedDict, namedtuple

from jax import grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _sig_slope

HMF_XTP = 13.0
HMF_K = 1.2
DEFAULT_HMF_KERN_PDICT = OrderedDict(ytp=-3.32, x0=14.0, lo=-0.95, hi=-1.8)
HMF_Params = namedtuple("HMF_Params", DEFAULT_HMF_KERN_PDICT.keys())
DEFAULT_HMF_KERN_PARAMS = HMF_Params(**DEFAULT_HMF_KERN_PDICT)


@jjit
def lg_hmf_kern(params, lgmp):
    params = HMF_Params(*params)
    lg_hmf = _sig_slope(
        lgmp, HMF_XTP, params.ytp, params.x0, HMF_K, params.lo, params.hi
    )
    return lg_hmf


@jjit
def hmf_kern(params, lgmu):
    lg_cuml = lg_hmf_kern(params, lgmu)
    return 10**lg_cuml


_differential_hmf_kern = jjit(vmap(grad(hmf_kern, argnums=1), in_axes=(None, 0)))


@jjit
def lg_differential_hmf_kern(params, lgmu):
    return jnp.log10(-_differential_hmf_kern(params, lgmu))
