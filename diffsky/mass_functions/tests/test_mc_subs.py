"""
"""
import numpy as np
from jax import random as jran

from ..ccshmf_model import DEFAULT_CCSHMF_PARAMS, predict_ccshmf
from ..mc_subs import (
    _compute_mean_subhalo_counts,
    generate_subhalopop,
    mc_generate_subhalopop_singlehalo,
)
from ..measure_ccshmf import NPTCL_CUT, SMDPL_LGMP, get_lgmu_cutoff

SMDPL_LGMP_CUT = SMDPL_LGMP + np.log10(NPTCL_CUT)


def infer_redshift_from_bname(bn):
    return float(bn.split("_")[4])


def infer_logmhost_from_bname(bn):
    return float(bn.split("_")[-1][:-4])


def _mae(pred, target):
    diff = pred - target
    return np.mean(np.abs(diff))


def test_mc_generate_subhalopop_singlehalo():
    lgmu_data = np.linspace(-6, 0, 100)
    ntot = int(1e5)
    ran_key = jran.PRNGKey(0)
    mc_lgmu = mc_generate_subhalopop_singlehalo(ran_key, lgmu_data, ntot)
    assert mc_lgmu.shape == (ntot,)
    assert np.all(np.isfinite(mc_lgmu))
    assert np.all(mc_lgmu >= lgmu_data.min())
    assert np.all(mc_lgmu <= lgmu_data.max())


def test_compute_mean_subhalo_counts():
    lgmp_cutoff = 11.0

    # subhalo counts should be strictly positive
    lgmhost = 12.0
    mean_counts = _compute_mean_subhalo_counts(lgmhost, lgmp_cutoff)
    assert mean_counts.shape == ()
    assert np.all(mean_counts > 0)

    # Decreasing the cutoff in μ should increase subhalo counts
    mean_counts2 = _compute_mean_subhalo_counts(lgmhost, lgmp_cutoff - 0.1)
    assert mean_counts.shape == ()
    assert np.all(mean_counts2 > mean_counts)

    # Increasing host halo mass should increase subhalo counts
    lgmhost_arr = np.linspace(10, 15, 100)
    mean_counts_arr = _compute_mean_subhalo_counts(lgmhost_arr, lgmp_cutoff)
    assert mean_counts_arr.shape == lgmhost_arr.shape
    assert np.all(mean_counts_arr > 0)
    assert np.all(np.diff(mean_counts_arr) > 0)


def test_generate_subhalopop_returns_reasonable_outputs():
    lgmp_cutoff = 11.0

    ran_key = jran.PRNGKey(0)
    nhosts = 10
    lgmhost_arr = np.linspace(12, 15, nhosts)
    res = generate_subhalopop(ran_key, lgmhost_arr, lgmp_cutoff)
    mc_lg_mu, lgmhost_pop, host_halo_indx = res
    for x in res:
        assert np.all(np.isfinite(x))
    assert np.all(mc_lg_mu <= 0)
    assert np.all(mc_lg_mu >= -10)

    # host_halo_indx array should provide the correct Mhost correspondence
    assert np.allclose(lgmhost_arr[host_halo_indx], lgmhost_pop)


def test_generate_subhalopop_has_expected_behavior():
    lgmp_cutoff = 11.0

    ran_key = jran.PRNGKey(0)
    nhosts = 200
    lgmhost_arr = np.repeat((12, 15), nhosts // 2)
    res = generate_subhalopop(ran_key, lgmhost_arr, lgmp_cutoff)
    mc_lg_mu, lgmhost_pop = res[:2]

    assert np.all(mc_lg_mu <= 0)

    # most subhalos will live in massive hosts
    assert np.allclose(lgmhost_pop.mean(), 15, atol=0.1)

    # Subhalos should not fall below the resolution limit for their host
    lgmu_cutoff_12 = get_lgmu_cutoff(12, lgmp_cutoff, 1)
    assert np.all(mc_lg_mu[lgmhost_pop == 12] >= lgmu_cutoff_12)

    lgmu_cutoff_15 = get_lgmu_cutoff(15, lgmp_cutoff, 1)
    assert np.all(mc_lg_mu[lgmhost_pop == 15] >= lgmu_cutoff_15)

    # Some subhalos in clusters should have lower μ than Milky Way subs
    assert lgmu_cutoff_15 < lgmu_cutoff_12
    assert np.any(mc_lg_mu[lgmhost_pop == 15] < lgmu_cutoff_12)


def test_generate_subhalopop_agrees_with_analytical_ccshmf():
    """This test ensures mutually consistent behavior between the generate_subhalopop
    Monte Carlo generator and the analytic function predict_ccshmf"""
    ran_key = jran.PRNGKey(0)

    nhosts_mc = 5_000

    lgmhost_targets = np.linspace(12, 15, 5)

    for itarget in range(lgmhost_targets.size):
        target_lgmhost = lgmhost_targets[itarget]
        lgmu_min = SMDPL_LGMP_CUT - target_lgmhost
        target_lgmu_bins = np.linspace(lgmu_min, -0.2, 20)
        pred_lg_ccshmf = predict_ccshmf(
            DEFAULT_CCSHMF_PARAMS, target_lgmhost, target_lgmu_bins
        )

        lgmhost_arr = np.zeros(nhosts_mc) + lgmhost_targets[itarget]
        ran_key, mc_key = jran.split(ran_key, 2)
        mc_lg_mu, __, __ = generate_subhalopop(mc_key, lgmhost_arr, SMDPL_LGMP_CUT)
        mc_lg_cuml_counts = np.log10(
            np.array([np.sum(mc_lg_mu > lgmu) for lgmu in target_lgmu_bins]) / nhosts_mc
        )
        loss_mae = _mae(pred_lg_ccshmf, mc_lg_cuml_counts)
        assert loss_mae < 0.04
