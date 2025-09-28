"""Utility functions for building PDF models"""

from functools import partial

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax.scipy import stats as jstats


@partial(jjit, static_argnames=["shape"])
def truncated_normal_sample(ran_key, shape, mu, sigma, x_min, x_max):
    """"""
    # Compute CDF values at truncation points
    alpha = (x_min - mu) / sigma
    beta = (x_max - mu) / sigma

    cdf_x_min = jstats.norm.cdf(alpha)
    cdf_x_max = jstats.norm.cdf(beta)

    uran = jran.uniform(ran_key, shape, minval=cdf_x_min, maxval=cdf_x_max)

    # Inverse CDF to get truncated normal samples
    z = jstats.norm.ppf(uran)
    x_sample = mu + sigma * z

    # Clamp to bounds for numerical safety
    x_sample = jnp.clip(x_sample, x_min, x_max)

    return x_sample
