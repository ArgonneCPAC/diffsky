""" """

import numpy as np
from jax import random as jran

from .. import disk_shapes as dshape


def test_sample_disk_axis_ratios():
    ran_key = jran.key(0)
    n_gals = 2_000
    axis_ratios = dshape.sample_disk_axis_ratios(
        ran_key, n_gals, dshape.DEFAULT_DISK_PARAMS
    )
    assert np.all(np.isfinite(axis_ratios.b_over_a))
    assert np.all(np.isfinite(axis_ratios.c_over_a))

    assert np.all(axis_ratios.b_over_a <= 1)
    assert np.any(axis_ratios.b_over_a < 1)
    assert np.all(axis_ratios.b_over_a > 0)

    assert np.all(axis_ratios.c_over_a <= 1)
    assert np.any(axis_ratios.c_over_a < 1)
    assert np.all(axis_ratios.c_over_a > 0)

    # Enforce reasonable diversity of shapes for default model
    assert np.std(axis_ratios.b_over_a) > 0.02
    assert np.std(axis_ratios.c_over_a) > 0.02
