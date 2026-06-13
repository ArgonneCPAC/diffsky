""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ....merging import merging_model
from ....param_utils import diffsky_param_wrapper_merging as dpwm
from ...tests import test_lightcone_generators as tlcg
from ...tests import test_mc_phot
from .. import phot_kernels


def test_mc_phot_kern(num_halos=75):
    ran_key = jran.key(0)
    # lc_data, tcurves = tmclh._get_weighted_lc_data_for_unit_testing(num_halos=num_halos)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )

    fb = 0.156
    ran_key, phot_key = jran.split(ran_key, 2)

    # Assume all galaxies are centrals so we can test against phot_kernels.py,
    # which does not implement satellite quenching
    n_gals = lc_data.z_obs.size
    upid = np.zeros(n_gals).astype(int) - 1
    lgmu_infall = np.zeros(n_gals).astype(int)
    logmhost_infall = np.zeros(n_gals).astype(int)
    gyr_since_infall = np.zeros(n_gals).astype(int)

    _res = phot_kernels._mc_phot_kern(
        phot_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpwm.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )
    mc_gd_phot_kern_results, mc_gd_phot_randoms, diffstarpop_results = _res

    test_mc_phot.check_phot_kern_results(mc_gd_phot_kern_results)

    # return mc_gd_phot_kern_results

    n_gals, n_bands = mc_gd_phot_kern_results.obs_mags.shape
    obs_mags_mc = np.zeros((n_gals, n_bands))
    obs_mags_mc = np.where(
        mc_gd_phot_kern_results.mc_sfh_type.reshape((n_gals, 1)) == 0,
        mc_gd_phot_kern_results.obs_mags_q,
        obs_mags_mc,
    )
    obs_mags_mc = np.where(
        mc_gd_phot_kern_results.mc_sfh_type.reshape((n_gals, 1)) == 1,
        mc_gd_phot_kern_results.obs_mags_ms,
        obs_mags_mc,
    )
    obs_mags_mc = np.where(
        mc_gd_phot_kern_results.mc_sfh_type.reshape((n_gals, 1)) == 2,
        mc_gd_phot_kern_results.obs_mags_bursty,
        obs_mags_mc,
    )
    assert np.allclose(obs_mags_mc, mc_gd_phot_kern_results.obs_mags, rtol=1e-5)

    t_infall = lc_data.t_obs - gyr_since_infall
    logmp_infall = lgmu_infall + logmhost_infall
    p_merge_smooth = merging_model.get_p_merge_from_merging_params(
        dpwm.DEFAULT_PARAM_COLLECTION.merging_params,
        logmp_infall,
        logmhost_infall,
        lc_data.t_obs,
        t_infall,
        upid,
    )

    gd_phot_kern_results = phot_kernels._phot_kern(
        mc_gd_phot_randoms,
        diffstarpop_results,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        p_merge_smooth,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        dpwm.DEFAULT_PARAM_COLLECTION.mzr_params,
        dpwm.DEFAULT_PARAM_COLLECTION.spspop_params,
        dpwm.DEFAULT_PARAM_COLLECTION.scatter_params,
        dpwm.DEFAULT_PARAM_COLLECTION.ssperr_params,
        DEFAULT_COSMOLOGY,
        fb,
    )

    skip_keys = (
        "burstiness_info_q",
        "burstiness_info_ms",
        "diffstar_info_ms",
        "diffstar_info_q",
    )
    for key in mc_gd_phot_kern_results._fields:
        if key not in skip_keys:
            x = getattr(mc_gd_phot_kern_results, key)
            x2 = getattr(gd_phot_kern_results, key)
            assert np.allclose(x, x2)


def test_mc_phot_kern_satellite_specific_effects(num_halos=75):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )

    fb = 0.156
    ran_key, phot_key = jran.split(ran_key, 2)

    # Assume all galaxies are centrals so we can test against phot_kernels.py,
    # which does not implement satellite quenching
    n_gals = lc_data.z_obs.size
    upid_cens = np.zeros(n_gals).astype(int) - 1
    lgmu_infall_cens = np.zeros(n_gals).astype(int)
    logmhost_infall_cens = np.zeros(n_gals).astype(int)
    gyr_since_infall_cens = np.zeros(n_gals).astype(int)

    _res = phot_kernels._mc_phot_kern(
        phot_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        upid_cens,
        lgmu_infall_cens,
        logmhost_infall_cens,
        gyr_since_infall_cens,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpwm.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )
    mc_gd_phot_kern_results_all_cens = _res[0]

    test_mc_phot.check_phot_kern_results(mc_gd_phot_kern_results_all_cens)

    upid = np.where(lc_data.is_central == 1, -1, lc_data.halo_indx)
    gyr_since_infall = lc_data.t_obs - lc_data.t_infall
    lgmu_infall = lc_data.logmp_infall - lc_data.logmhost_infall

    _res = phot_kernels._mc_phot_kern(
        phot_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        upid,
        lgmu_infall,
        lc_data.logmhost_infall,
        gyr_since_infall,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpwm.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )
    mc_gd_phot_kern_results_with_sats = _res[0]
    test_mc_phot.check_phot_kern_results(mc_gd_phot_kern_results_with_sats)

    assert not np.allclose(
        mc_gd_phot_kern_results_with_sats.obs_mags,
        mc_gd_phot_kern_results_all_cens.obs_mags,
        rtol=1e-3,
    )
