""""""

import numpy as np
import pytest
from jax import random as jran

from .. import disk_bulge_sizes as dbs
from .. import smzr_bulge

try:
    from scipy.stats import binned_statistic

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _mae(y1, y2):
    diff = y2 - y1
    return np.mean(np.abs(diff))


def test_lgr50_kern_bulge_default_model():
    n = 100
    marr = np.logspace(8.0, 11.5, n)
    lgmarr = np.log10(marr)
    ZZ = np.zeros(n)

    Z_TABLE = (0, 0.5, 1, 2)
    for redshift in Z_TABLE:
        lgr50_ek = np.log10(dbs._bulge_median_r50(marr, ZZ + redshift))
        lgr50_sig_slope = smzr_bulge._lgr50_kern_bulge(
            lgmarr, ZZ + redshift, smzr_bulge._DBS_BULGE_SIZE_PARAMS
        )
        mae_diff = _mae(lgr50_ek, lgr50_sig_slope)
        assert mae_diff < 0.15


def test_mc_r50_bulge_size_behaves_as_expected():
    """Check the returned sizes are within expected bounds"""
    ran_key = jran.key(0)
    n_gals = 20_000
    logsmarr = np.linspace(5, 13, n_gals)
    zarr = np.zeros(n_gals)
    r50_galpop, zscore = smzr_bulge.mc_r50_bulge_size(logsmarr, zarr, ran_key)

    assert r50_galpop.shape == (n_gals,)
    assert np.all(np.isfinite(r50_galpop))
    assert np.all(np.isfinite(zscore))
    assert np.allclose(zscore.mean(), 0.0, atol=0.1)

    assert np.all(r50_galpop > 0)
    assert np.any(r50_galpop > smzr_bulge.R50_MAX)

    frac_larger_than_max = np.mean(r50_galpop > smzr_bulge.R50_MAX)
    assert frac_larger_than_max < 0.02

    frac_smaller_than_min = np.mean(r50_galpop < smzr_bulge.R50_MIN)
    assert frac_smaller_than_min < 0.02


@pytest.mark.skipif(not HAS_SCIPY, reason="Must have scipy installed to run this test")
def test_mc_r50_bulge_size_consistent_with_analytic_expectation():
    """Compare median of Monte Carlo realization to analytic median"""
    ran_key = jran.key(0)
    n_gals = 250_000
    logsmarr = np.linspace(5, 13, n_gals)
    zarr = np.zeros(n_gals)
    r50_galpop, zscore = smzr_bulge.mc_r50_bulge_size(logsmarr, zarr, ran_key)

    logsm_bins = np.linspace(logsmarr.min(), logsmarr.max(), 30)
    median_r50_measured, __, __ = binned_statistic(
        logsmarr, r50_galpop, bins=logsm_bins, statistic="median"
    )
    median_r50_analytic = 10 ** smzr_bulge._lgr50_kern_bulge(
        logsmarr, zarr, smzr_bulge.DEFAULT_BULGE_SIZE_PARAMS
    )
    logsm_binmids = 0.5 * (logsm_bins[:-1] + logsm_bins[1:])
    median_r50_analytic_interp = np.interp(logsm_binmids, logsmarr, median_r50_analytic)
    mae_loss = _mae(median_r50_measured, median_r50_analytic_interp)
    assert mae_loss < 0.05
