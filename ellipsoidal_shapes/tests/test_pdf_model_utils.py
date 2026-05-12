""" """

import numpy as np
from jax import random as jran

from .. import pdf_model_utils as pmu


def test_truncated_normal_sample():
    ran_key = jran.key(0)
    n_gals = 2_000
    mu = 0.4
    sigma = 0.2
    x_min, x_max = 0.2, 1.0
    x_sample, zscore = pmu.truncated_normal_sample(
        ran_key, (n_gals,), mu, sigma, x_min, x_max
    )
    assert np.all(np.isfinite(x_sample))
    assert np.all(x_sample >= x_min)
    assert np.all(x_sample <= x_max)
    assert np.all(np.isfinite(zscore))
    assert zscore.shape == x_sample.shape
