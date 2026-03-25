""""""

from jax import jit as jjit
from jax import numpy as jnp

from ...utils import _sigmoid

FDD_MIN = 0.1
FDD_MAX = 0.9


@jjit
def _frac_disk_dom_kern(logsm, logssfr):
    delta_fdd = _delta_fdd_ssfr(logssfr)
    ylo = jnp.clip(0.9 + delta_fdd, min=FDD_MIN, max=FDD_MAX)
    yhi = jnp.clip(0.1 + delta_fdd, min=FDD_MIN, max=FDD_MAX)
    return _sigmoid(logsm, 10.75, 1.5, ylo, yhi)


@jjit
def _delta_fdd_ssfr(logssfr):
    delta_fdd = _sigmoid(logssfr, -10.5, 2.0, -0.25, 0.25)
    return delta_fdd
