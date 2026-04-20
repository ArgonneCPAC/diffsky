""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ....merging import merging_model
from ....param_utils import diffsky_param_wrapper as dpw
from ...tests import test_lightcone_generators as tlcg
from .. import phot_kernels_merging as pkm


def test_mc_phot_kern_merging(num_halos=250):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.176

    mc_merge = 0
    _res = pkm._mc_phot_kern_merging(
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
    phot_kern_results, phot_randoms, flux_obs, merge_prob, mstar_obs = _res
    assert np.all(merge_prob >= 0)
    assert np.all(merge_prob <= 1)
    assert np.any(merge_prob > 0)
    assert np.any(merge_prob < 1)

    assert np.all(np.isfinite(mstar_obs))

    obs_mags_in_plus_ex_situ = -2.5 * np.log10(flux_obs)
    assert np.any(obs_mags_in_plus_ex_situ != phot_kern_results.obs_mags)

    # Enforce centrals get brighter and satellites get dimmer
    assert np.all(
        obs_mags_in_plus_ex_situ[lc_data.is_central == 1]
        <= phot_kern_results.obs_mags[lc_data.is_central == 1]
    )
    assert np.all(
        obs_mags_in_plus_ex_situ[lc_data.is_central == 0]
        >= phot_kern_results.obs_mags[lc_data.is_central == 0]
    )

    # Enforce centrals get more massive and satellites less massive
    assert np.all(
        mstar_obs[lc_data.is_central == 1]
        >= 10 ** phot_kern_results.logsm_obs[lc_data.is_central == 1]
    )
    assert np.all(
        mstar_obs[lc_data.is_central == 0]
        <= 10 ** phot_kern_results.logsm_obs[lc_data.is_central == 0]
    )
