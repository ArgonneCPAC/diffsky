""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ....merging import merging_model
from ....param_utils import diffsky_param_wrapper as dpw
from ...tests import test_lightcone_generators as tlcg
from .. import phot_kernels_merging as pkm


def check_phot_kern_merging_results(phot_kern_results, lc_data):
    n_gals = lc_data.z_obs.size
    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape

    for arr in phot_kern_results:
        assert np.all(np.isfinite(arr))

    assert np.all(phot_kern_results.p_merge >= 0)
    assert np.all(phot_kern_results.p_merge <= 1)
    assert np.any(phot_kern_results.p_merge > 0)
    assert np.any(phot_kern_results.p_merge < 1)

    # Enforce consistent array shapes
    assert phot_kern_results.p_merge.shape == (n_gals,)
    assert phot_kern_results.logsm_obs.shape == (n_gals,)
    assert phot_kern_results.logsm_obs_in_situ.shape == (n_gals,)
    assert phot_kern_results.obs_mags.shape == (n_gals, n_bands)
    assert phot_kern_results.obs_mags_in_situ.shape == (n_gals, n_bands)

    msk_cen = lc_data.is_central == 1
    msk_sat = ~msk_cen

    # Enforce centrals can only get brighter and satellites can only get dimmer
    name = "obs_mags"
    x = getattr(phot_kern_results, name)
    y = getattr(phot_kern_results, name + "_in_situ")
    assert np.all(x[msk_cen] <= y[msk_cen])
    assert np.all(x[msk_sat] >= y[msk_sat])
    # Enforce merging is nontrivial
    assert np.any(x[msk_cen] < y[msk_cen])
    assert np.any(x[msk_sat] > y[msk_sat])

    # Enforce centrals can only get more massive and satellites less massive
    name = "logsm_obs"
    x = getattr(phot_kern_results, name)
    y = getattr(phot_kern_results, name + "_in_situ")
    assert np.all(x[msk_cen] >= y[msk_cen])
    assert np.all(x[msk_sat] <= y[msk_sat])
    # Enforce merging is nontrivial
    assert np.any(x[msk_cen] > y[msk_cen])
    assert np.any(x[msk_sat] < y[msk_sat])


def test_mc_phot_kern_merging(num_halos=250):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.176

    mc_merge = 0
    phot_kern_results, phot_randoms = pkm._mc_phot_kern_merging(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpw.DEFAULT_PARAM_COLLECTION,
        merging_model.DEFAULT_MERGE_PARAMS,
        DEFAULT_COSMOLOGY,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.nhalos,
        lc_data.halo_indx,
        mc_merge,
    )

    check_phot_kern_merging_results(phot_kern_results, lc_data)
