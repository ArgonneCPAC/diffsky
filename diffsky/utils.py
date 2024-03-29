"""
"""

from jax import jit as jjit
from jax import lax, nn


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff * nn.sigmoid(k * (x - x0))


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - lax.log(lnarg) / k
