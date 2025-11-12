"""Ellipsoidal model for PDF of disk shapes"""

from collections import namedtuple
from functools import partial

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

AxisRatios = namedtuple("AxisRatios", ("b_over_a", "c_over_a"))
DiskAxisRatioParams = namedtuple(
    "DiskAxisRatioParams", ("ba_min", "ba_max", "c_min", "c_max")
)
DEFAULT_DISK_PARAMS = DiskAxisRatioParams(ba_min=0.8, ba_max=1.0, c_min=0.2, c_max=0.5)


@partial(jjit, static_argnames=["n_samples"])
def sample_disk_axis_ratios(ran_key, n_samples, disk_params=DEFAULT_DISK_PARAMS):
    """Draw samples of axis ratios for disks"""
    ran_key_b, ran_key_c = jran.split(ran_key, 2)

    b_over_a = jran.uniform(
        ran_key_b, (n_samples,), minval=disk_params.ba_min, maxval=disk_params.ba_max
    )

    c_over_b = jran.uniform(
        ran_key_c, (n_samples,), minval=disk_params.c_min, maxval=disk_params.c_max
    )

    c_over_a = c_over_b * b_over_a
    c_over_a = jnp.minimum(c_over_a, b_over_a * 0.99)  # Ensure c <= b for safety

    axis_ratios = AxisRatios(b_over_a, c_over_a)
    return axis_ratios
