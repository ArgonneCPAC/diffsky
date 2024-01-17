"""The lg_ccshmf_kern function gives a differentiable prediction for
the Cumulative Conditional Subhalo Mass Function (CCSHMF),
<Nsub(>μ) | Mhost>, where μ = Msub/Mhost.
The lg_ccshmf_kern function is a single-halo kernel called by ccshmf.predict_ccshmf

"""
from collections import OrderedDict, namedtuple

from jax import grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _sig_slope

XTP = -1.0
K = 10
X0 = -0.35
YHI = -3.1

DEFAULT_CCSHMF_KERN_PDICT = OrderedDict(ytp=-0.3, ylo=-0.95)
CCSHMF_Params = namedtuple("CCSHMF_Params", DEFAULT_CCSHMF_KERN_PDICT.keys())
DEFAULT_CCSHMF_KERN_PARAMS = CCSHMF_Params(**DEFAULT_CCSHMF_KERN_PDICT)


@jjit
def lg_ccshmf_kern(params, lgmu):
    params = CCSHMF_Params(*params)
    lg_ccshmf = _sig_slope(lgmu, XTP, params.ytp, X0, K, params.ylo, YHI)
    return lg_ccshmf


@jjit
def ccshmf_kern(params, lgmu):
    lg_cuml = lg_ccshmf_kern(params, lgmu)
    return 10**lg_cuml


_differential_cshmf_kern = jjit(vmap(grad(ccshmf_kern, argnums=1), in_axes=(None, 0)))


@jjit
def lg_differential_cshmf_kern(params, lgmu):
    return jnp.log10(-_differential_cshmf_kern(params, lgmu))
