"""
Test of zhang_yang sizes
"""
import numpy as np

from ..zhang_yang17 import (
    mc_size_vs_luminosity_early_type,
    mc_size_vs_luminosity_late_type,
)

NGALS_TEST = int(1e7)
SIZE_MAX = 101.0  # kpc


def test_mc_size_vs_luminosity_early_type():
    magr = np.random.uniform(-100, 100, NGALS_TEST)
    redshift = np.random.uniform(0, 20, NGALS_TEST)
    size = mc_size_vs_luminosity_early_type(magr, redshift)
    assert np.all(size < SIZE_MAX)


def test_mc_size_vs_luminosity_late_type():
    magr = np.random.uniform(-100, 100, NGALS_TEST)
    redshift = np.random.uniform(0, 20, NGALS_TEST)
    size = mc_size_vs_luminosity_late_type(magr, redshift)
    assert np.all(size < SIZE_MAX)
