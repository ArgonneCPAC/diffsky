""" """

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_emline_info as lemi
from jax import random as jran

from ....merging import merging_model
from ....param_utils import diffsky_param_wrapper as dpw
from ...tests import test_lightcone_generators as tlcg
from .. import mc_randoms
from .. import specphot_kernels_merging as sppkm


def test_mc_specphot_kern_merging(num_halos=41):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.151

    n_lines = 3
    line_wave_table = np.linspace(1_000, 10_000, n_lines)
    emline_names = lc_data.ssp_data.ssp_emline_wave._fields[0:n_lines]
    ssp_data = lemi.get_subset_emline_data(lc_data.ssp_data, emline_names)
    lc_data = lc_data._replace(ssp_data=ssp_data)

    mc_merge = 0
    _res = sppkm._mc_specphot_kern_merging(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        line_wave_table,
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
    (
        phot_kern_results,
        linelums_in_situ,
        phot_randoms,
        flux_obs,
        merge_prob,
        mstar_obs,
        linelums_in_plus_ex_situ,
    ) = _res
    assert np.all(merge_prob >= 0)
    assert np.all(merge_prob <= 1)
    assert np.any(merge_prob > 0)
    assert np.any(merge_prob < 1)

    assert np.all(np.isfinite(mstar_obs))

    obs_mags_in_plus_ex_situ = -2.5 * np.log10(flux_obs)
    assert np.any(obs_mags_in_plus_ex_situ != phot_kern_results.obs_mags)
    assert np.any(linelums_in_plus_ex_situ != linelums_in_situ)

    # Enforce centrals can only get brighter and satellites can only get dimmer
    assert np.all(
        obs_mags_in_plus_ex_situ[lc_data.is_central == 1]
        <= phot_kern_results.obs_mags[lc_data.is_central == 1]
    )
    assert np.all(
        obs_mags_in_plus_ex_situ[lc_data.is_central == 0]
        >= phot_kern_results.obs_mags[lc_data.is_central == 0]
    )

    # Enforce centrals can only get more massive and satellites less massive
    assert np.all(
        mstar_obs[lc_data.is_central == 1]
        >= 10 ** phot_kern_results.logsm_obs[lc_data.is_central == 1]
    )
    assert np.all(
        mstar_obs[lc_data.is_central == 0]
        <= 10 ** phot_kern_results.logsm_obs[lc_data.is_central == 0]
    )

    # Enforce centrals can only get brighter lines and satellites less bright
    assert np.all(
        linelums_in_plus_ex_situ[lc_data.is_central == 1]
        >= linelums_in_situ[lc_data.is_central == 1]
    )
    assert np.all(
        linelums_in_plus_ex_situ[lc_data.is_central == 0]
        <= linelums_in_situ[lc_data.is_central == 0]
    )

    # Enforce some centrals actually get brighter and some satellites dimmer
    assert np.any(
        obs_mags_in_plus_ex_situ[lc_data.is_central == 1]
        < phot_kern_results.obs_mags[lc_data.is_central == 1]
    )
    assert np.any(
        obs_mags_in_plus_ex_situ[lc_data.is_central == 0]
        > phot_kern_results.obs_mags[lc_data.is_central == 0]
    )

    # Enforce some centrals actually get more massive and satellites less massive
    assert np.any(
        mstar_obs[lc_data.is_central == 1]
        > 10 ** phot_kern_results.logsm_obs[lc_data.is_central == 1]
    )
    assert np.any(
        mstar_obs[lc_data.is_central == 0]
        < 10 ** phot_kern_results.logsm_obs[lc_data.is_central == 0]
    )

    # Enforce some centrals actually get brighter lines and satellites less bright
    assert np.any(
        linelums_in_plus_ex_situ[lc_data.is_central == 1]
        > linelums_in_situ[lc_data.is_central == 1]
    )
    assert np.any(
        linelums_in_plus_ex_situ[lc_data.is_central == 0]
        < linelums_in_situ[lc_data.is_central == 0]
    )


def test_specphot_kern_merging(num_halos=47):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )

    fb = 0.116
    ran_key, phot_key = jran.split(ran_key, 2)

    phot_randoms, sfh_params, merging_randoms = mc_randoms.get_mc_phot_merge_randoms(
        ran_key, dpw.DEFAULT_PARAM_COLLECTION[0], lc_data.mah_params, DEFAULT_COSMOLOGY
    )

    n_lines = 3
    line_wave_table = np.linspace(1_000, 10_000, n_lines)
    emline_names = lc_data.ssp_data.ssp_emline_wave._fields[0:n_lines]
    ssp_data = lemi.get_subset_emline_data(lc_data.ssp_data, emline_names)
    lc_data = lc_data._replace(ssp_data=ssp_data)

    mc_merge = 0

    (
        phot_kern_results,
        linelums_in_situ,
        flux_obs,
        merge_prob,
        mstar_obs,
        linelums_in_plus_ex_situ,
    ) = sppkm._specphot_kern_merging(
        phot_randoms,
        merging_randoms,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        line_wave_table,
        *dpw.DEFAULT_PARAM_COLLECTION[1:],
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

    assert np.all(merge_prob >= 0)
    assert np.all(merge_prob <= 1)
    assert np.any(merge_prob > 0)
    assert np.any(merge_prob < 1)

    assert np.all(np.isfinite(mstar_obs))

    assert np.any(linelums_in_plus_ex_situ != linelums_in_situ)
