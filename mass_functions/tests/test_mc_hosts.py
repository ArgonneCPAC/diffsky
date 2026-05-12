""" """

import numpy as np
from jax import random as jran

from ..hmf_model import DEFAULT_HMF_PARAMS, predict_differential_hmf
from ..mc_hosts import LGMH_MAX, mc_host_halos_singlez


def test_mc_host_halo_logmp_behaves_as_expected():
    ran_key = jran.PRNGKey(0)

    lgmp_min, redshift, volume_com = 11.0, 0.1, 100**3
    lgmp_halopop = mc_host_halos_singlez(ran_key, lgmp_min, redshift, volume_com)
    assert lgmp_halopop.size > 0
    assert np.all(lgmp_halopop > lgmp_min)
    assert np.all(lgmp_halopop < LGMH_MAX)

    lgmp_max = 14.0
    lgmp_halopop2 = mc_host_halos_singlez(
        ran_key, lgmp_min, redshift, volume_com, lgmp_max=lgmp_max
    )
    assert np.all(lgmp_halopop > lgmp_min)
    assert np.all(lgmp_halopop2 < lgmp_max)


def test_differential_cumulative_hmf_consistency():
    ran_key = jran.key(0)
    z = 0.0
    lgmp_min = 12.0
    Lbox = 1000.0
    Vbox = Lbox**3

    lgm_halopop = mc_host_halos_singlez(ran_key, lgmp_min, z, Vbox)
    lgm_bins = np.linspace(lgmp_min + 0.5, 14.0, 50)

    differential_hmf = predict_differential_hmf(DEFAULT_HMF_PARAMS, lgm_bins, z)

    binned_counts = np.histogram(lgm_halopop, bins=lgm_bins)[0]
    differential_hmf_target = binned_counts / Vbox / np.diff(lgm_bins)

    # Interpolate to compare same-sized arrays
    lgm_binmids = 0.5 * (lgm_bins[:-1] + lgm_bins[1:])
    lg_diff_hmf = np.log10(differential_hmf)
    diff_hmf_interp = 10 ** np.interp(lgm_binmids, lgm_bins, lg_diff_hmf)

    assert np.allclose(diff_hmf_interp, differential_hmf_target, rtol=0.1)
