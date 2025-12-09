""""""

import random
from collections import namedtuple

import numpy as np
import pytest
from jax import random as jran
from jax.scipy.stats import norm


def get_random_transmission_curve(
    ran_key=None, tcurve_center=None, wave_range=(1000, 10_000), scale=300
):
    if ran_key is None:
        seed = random.randint(0, 2**32 - 1)
        ran_key = jran.key(seed)

    if tcurve_center is None:
        xmin = wave_range[0] + scale * 2
        xmax = wave_range[1] - scale * 2
        tcurve_center = jran.uniform(ran_key, minval=xmin, maxval=xmax)
    else:
        assert wave_range[0] < tcurve_center < wave_range[1]

    wave = np.linspace(*wave_range, 200)

    _transmission = norm.pdf(wave, loc=tcurve_center, scale=scale)
    transmission = _transmission / _transmission.max()
    TransmissionCurve = namedtuple("TransmissionCurve", ("wave", "transmission"))
    tcurve = TransmissionCurve(wave, transmission)

    return tcurve


@pytest.fixture(scope="module")
def lc_data():
    """Generate lightcone data once per test module."""
    return _generate_testing_lightcone()


def _generate_testing_lightcone():
    return (1, 2, 3)


def test1(lc_data):
    pass


def test2():
    pass


def test3():
    pass
