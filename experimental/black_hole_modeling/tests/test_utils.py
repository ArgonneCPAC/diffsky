""" """

import numpy as np

from .. import utils


def test_approximate_ssfr_percentile():
    ssfr = np.linspace(-15, 15, 1_000)
    p = utils.approximate_ssfr_percentile(ssfr)
    assert np.all(np.isfinite(p))
    assert np.all(p > 0)
    assert np.all(p < 1)
