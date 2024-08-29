"""
"""

import numpy as np

from .. import smhm


def test_compute_smhm():
    n_halos = 2_000
    logmh = np.linspace(10, 15, n_halos)
    logsm = logmh - 2.0

    n_bins = 20
    logmh_bins = np.linspace(11, 14, n_bins)
    sigma = np.zeros(n_halos) + np.mean(np.diff(logmh_bins)) / 2
    mean_logsm = smhm.compute_smhm(logmh, logsm, sigma, logmh_bins)
    assert mean_logsm.shape == (n_bins - 1,)
    assert np.all(np.isfinite(mean_logsm))
