"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from jax import lax


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff * lax.logistic(k * (x - x0))


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - jnp.log(lnarg) / k
