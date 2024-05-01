"""
"""

import numpy as np
from jax import random as jran

from ..dustea import AvBOUNDS, BumpBOUNDS, DeltaBOUNDS, att_curve


def test_tea_att_curve():
    n_wave = 500
    wave = np.linspace(1_000, 10_000, n_wave)

    ran_key = jran.PRNGKey(0)
    n_tests = 100
    for __ in range(n_tests):
        ran_key, av_key, delta_key, bump_key = jran.split(ran_key, 4)
        Av = jran.uniform(av_key, minval=AvBOUNDS[0], maxval=AvBOUNDS[1], shape=())
        delta = jran.uniform(
            bump_key, minval=DeltaBOUNDS[0], maxval=DeltaBOUNDS[1], shape=()
        )
        bump = jran.uniform(
            delta_key, minval=BumpBOUNDS[0], maxval=BumpBOUNDS[1], shape=()
        )
        att = att_curve(wave, Av, delta, bump)

        assert np.all(np.isfinite(att))
        assert np.all(att >= 0), (float(Av), float(delta), float(bump))
