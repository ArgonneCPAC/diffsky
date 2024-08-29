"""
"""

import numpy as np

from ..measure_hmf import measure_smdpl_lg_cuml_hmf


def test_measure_hmf_returns_finite():
    nhalos = int(1e6)
    nbins = 50
    logmp_data = np.linspace(10, 15, nhalos)

    logmp_bins = np.linspace(11, 14.5, nbins)
    logmp_target, lg_cuml_target = measure_smdpl_lg_cuml_hmf(logmp_data, logmp_bins)
    assert np.all(np.isin(logmp_target, logmp_bins))
    assert lg_cuml_target.shape == logmp_target.shape
    assert np.all(np.diff(lg_cuml_target) <= 0)
    assert np.any(np.diff(lg_cuml_target) < 0)

    logmp_bins = np.linspace(8, 14.5, nbins)
    logmp_target, lg_cuml_target = measure_smdpl_lg_cuml_hmf(logmp_data, logmp_bins)
    assert np.all(np.isin(logmp_target, logmp_bins))
    assert lg_cuml_target.shape == logmp_target.shape
    assert np.all(np.diff(lg_cuml_target) <= 0)
    assert np.any(np.diff(lg_cuml_target) < 0)

    # default bins
    logmp_target, lg_cuml_target = measure_smdpl_lg_cuml_hmf(logmp_data)
    assert lg_cuml_target.shape == logmp_target.shape
    assert np.all(np.diff(lg_cuml_target) <= 0)
    assert np.any(np.diff(lg_cuml_target) < 0)
