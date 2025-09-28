"""Ellipsoidal model for PDF of bulge shapes"""

from collections import namedtuple
from functools import partial

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from .pdf_model_utils import truncated_normal_sample

AxisRatios = namedtuple("AxisRatios", ("b_over_a", "c_over_a"))
BulgeAxisRatioParams = namedtuple(
    "BulgeAxisRatioParams", ("ba_peak", "ba_sigma", "ba_min", "ba_max", "c_min")
)
DEFAULT_BULGE_PARAMS = BulgeAxisRatioParams(
    ba_peak=1.3, ba_sigma=0.38, ba_min=0.3, ba_max=1.0, c_min=0.61
)


@partial(jjit, static_argnames=["n_samples"])
def sample_bulge_axis_ratios(ran_key, n_samples, bulge_params):
    """Draw samples of axis ratios for bulges"""
    ran_key_b, ran_key_c = jran.split(ran_key, 2)

    b_over_a = truncated_normal_sample(
        ran_key_b,
        (n_samples,),
        bulge_params.ba_peak,
        bulge_params.ba_sigma,
        bulge_params.ba_min,
        bulge_params.ba_max,
    )

    c_over_b = jran.uniform(
        ran_key_c, (n_samples,), minval=bulge_params.c_min, maxval=1.0
    )

    c_over_a = c_over_b * b_over_a
    c_over_a = jnp.minimum(c_over_a, b_over_a * 0.99)  # Ensure c <= b for safety

    axis_ratios = AxisRatios(b_over_a, c_over_a)
    return axis_ratios
