""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import random as jran

from ....merging import merging_model
from ....param_utils import diffsky_param_wrapper_merging as dpwm
from ...disk_bulge_modeling import dbpop
from ...tests import test_lightcone_generators as tlcg
from ...tests import test_mc_phot
from .. import gd_dbk_kernels
from .. import gd_dbk_specphot_kernels as gd_dbkspk
from .. import gd_phot_kernels, mc_randoms


def test_mc_dbk_kern(num_halos=50):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )

    fb = 0.156
    ran_key, phot_key, dbk_key = jran.split(ran_key, 3)
    n_gals = lc_data.z_obs.size
    dbk_randoms = mc_randoms.get_mc_dbk_randoms(dbk_key, n_gals)

    upid = np.where(lc_data.is_central == 1, -1, lc_data.halo_indx)
    gyr_since_infall = lc_data.t_obs - lc_data.t_infall
    lgmu_infall = lc_data.logmp_infall - lc_data.logmhost_infall
    phot_kern_results, phot_randoms, diffstarpop_results = (
        gd_phot_kernels._mc_phot_kern(
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
    )

    test_mc_phot.check_phot_kern_results(phot_kern_results)

    ran_key, dbk_key = jran.split(ran_key, 2)
    burst_params = DEFAULT_BURST_PARAMS._replace(
        lgfburst=phot_kern_results.lgfburst,
        lgyr_peak=phot_kern_results.lgyr_peak,
        lgyr_max=phot_kern_results.lgyr_max,
    )
    age_weights = np.sum(phot_kern_results.ssp_weights, axis=1)
    p_merge_smooth = merging_model.get_p_merge_from_merging_params(
        dpwm.DEFAULT_PARAM_COLLECTION.merging_params,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_obs,
        lc_data.t_infall,
        upid,
    )

    args = (
        lc_data.t_obs,
        lc_data.ssp_data,
        phot_kern_results.t_table,
        phot_kern_results.sfh_table,
        burst_params,
        phot_kern_results.lgmet_weights,
        dbk_randoms,
        phot_kern_results.logsm_obs,
        age_weights,
        p_merge_smooth,
    )
    dbk_weights, disk_bulge_history = gd_dbk_kernels._dbk_kern(*args)
    assert np.all(np.isfinite(dbk_weights.ssp_weights_bulge))
    assert np.all(np.isfinite(dbk_weights.ssp_weights_disk))
    assert np.all(np.isfinite(dbk_weights.ssp_weights_knots))

    assert np.allclose(
        np.sum(dbk_weights.ssp_weights_bulge, axis=(1, 2)), 1.0, rtol=1e-4
    )
    assert np.allclose(
        np.sum(dbk_weights.ssp_weights_disk, axis=(1, 2)), 1.0, rtol=1e-4
    )
    assert np.allclose(
        np.sum(dbk_weights.ssp_weights_knots, axis=(1, 2)), 1.0, rtol=1e-4
    )

    assert np.all(dbk_weights.mstar_bulge > 0)
    assert np.all(dbk_weights.mstar_disk > 0)
    assert np.all(dbk_weights.mstar_knots > 0)

    correct_shape = phot_kern_results.logsm_obs.shape
    assert dbk_weights.mstar_bulge.shape == correct_shape
    assert dbk_weights.mstar_disk.shape == correct_shape
    assert dbk_weights.mstar_knots.shape == correct_shape

    args = (
        phot_kern_results.ssp_photflux_table,
        dbk_weights,
        phot_kern_results.dust_frac_trans,
        phot_kern_results.frac_ssp_errors,
    )
    _res = gd_dbk_kernels._get_dbk_phot_from_dbk_weights(*args)
    obs_mags_bulge, obs_mags_disk, obs_mags_knots = _res

    correct_shape = phot_kern_results.obs_mags.shape
    assert obs_mags_bulge.shape == correct_shape
    assert obs_mags_disk.shape == correct_shape
    assert obs_mags_knots.shape == correct_shape

    np.all(phot_kern_results.logsm_obs > np.log10(dbk_weights.mstar_bulge.flatten()))
    np.all(phot_kern_results.logsm_obs > np.log10(dbk_weights.mstar_disk.flatten()))
    np.all(phot_kern_results.logsm_obs > np.log10(dbk_weights.mstar_knots.flatten()))

    assert np.all(np.isfinite(obs_mags_bulge))
    assert np.all(np.isfinite(obs_mags_disk))
    assert np.all(np.isfinite(obs_mags_knots))

    assert not np.allclose(phot_kern_results.obs_mags, obs_mags_bulge, rtol=1e-4)
    assert np.all(phot_kern_results.obs_mags <= obs_mags_bulge)

    assert not np.allclose(phot_kern_results.obs_mags, obs_mags_disk, rtol=1e-4)
    assert np.all(phot_kern_results.obs_mags <= obs_mags_disk)

    assert not np.allclose(phot_kern_results.obs_mags, obs_mags_knots, rtol=1e-4)
    assert np.all(phot_kern_results.obs_mags <= obs_mags_knots)

    a = 10 ** (-0.4 * obs_mags_bulge)
    b = 10 ** (-0.4 * obs_mags_disk)
    c = 10 ** (-0.4 * obs_mags_knots)
    mtot = -2.5 * np.log10(a + b + c)

    magdiff = mtot - phot_kern_results.obs_mags
    assert np.all(np.abs(magdiff) < 0.1)

    mean_magdiff = np.mean(magdiff, axis=0)  # shape = (n_bands,)
    assert np.allclose(mean_magdiff, 0.0, atol=0.01)

    std_magdiff = np.std(magdiff, axis=0)
    assert np.all(std_magdiff < 0.01)


def test_get_dbk_weights(num_halos=25):
    """Enforce all dbk_weights sum to unity and dbk masses sum to total mass"""
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )

    fb = 0.1
    upid = np.where(lc_data.is_central == 1, -1, lc_data.halo_indx)
    gyr_since_infall = lc_data.t_obs - lc_data.t_infall
    lgmu_infall = lc_data.logmp_infall - lc_data.logmhost_infall

    _res = gd_phot_kernels._mc_phot_kern(
        ran_key,
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
    phot_kern_results, phot_randoms, diffstarpop_results = _res
    dbk_key = jran.key(1)
    n_gals = lc_data.z_obs.size
    dbk_randoms = mc_randoms.get_mc_dbk_randoms(dbk_key, n_gals)

    disk_bulge_history = dbpop.decompose_sfh_into_disk_bulge_sfh(
        dbk_randoms.uran_fbulge,
        phot_kern_results.t_table,
        phot_kern_results.sfh_table,
        lc_data.t_obs,
    )
    burst_params = DEFAULT_BURST_PARAMS._replace(
        lgfburst=phot_kern_results.lgfburst,
        lgyr_peak=phot_kern_results.lgyr_peak,
        lgyr_max=phot_kern_results.lgyr_max,
    )

    age_weights = np.sum(phot_kern_results.ssp_weights, axis=1)
    p_merge_smooth = merging_model.get_p_merge_from_merging_params(
        dpwm.DEFAULT_PARAM_COLLECTION.merging_params,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_obs,
        lc_data.t_infall,
        upid,
    )

    dbk_weights = gd_dbk_kernels.get_dbk_weights_rq(
        lc_data.t_obs,
        lc_data.ssp_data,
        phot_kern_results.t_table,
        burst_params,
        phot_kern_results.lgmet_weights,
        disk_bulge_history,
        dbk_randoms.fknot,
        phot_kern_results.logsm_obs,
        age_weights,
        p_merge_smooth,
    )

    assert dbk_weights.mstar_bulge.shape == (n_gals,)
    assert dbk_weights.mstar_disk.shape == (n_gals,)
    assert dbk_weights.mstar_knots.shape == (n_gals,)
    logsm_sum = np.log10(
        dbk_weights.mstar_bulge + dbk_weights.mstar_disk + dbk_weights.mstar_knots
    )
    assert np.allclose(phot_kern_results.logsm_obs, logsm_sum, atol=0.1)
    n_gals, n_met, n_age = dbk_weights.ssp_weights_bulge.shape

    for weights in (
        dbk_weights.ssp_weights_bulge,
        dbk_weights.ssp_weights_disk,
        dbk_weights.ssp_weights_knots,
    ):
        assert weights.shape == (n_gals, n_met, n_age)
        assert np.allclose(np.sum(weights, axis=(1, 2)), 1.0, rtol=1e-3)


def test_get_dbk_linelum_decomposition(num_halos=55, n_lines=4):
    """Enforce that the sum of the component lines equals the composite line"""
    ran_key = jran.key(10)

    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos, n_lines=n_lines
    )
    fb = 0.196
    n_gals = lc_data.z_obs.size

    upid = np.where(lc_data.is_central == 1, -1, lc_data.halo_indx)
    gyr_since_infall = lc_data.t_obs - lc_data.t_infall
    lgmu_infall = lc_data.logmp_infall - lc_data.logmhost_infall

    args = (
        ran_key,
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
        lc_data.line_wave_table,
        *dpwm.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )
    dbk_specphot_info, dbk_weights = gd_dbkspk._mc_dbk_specphot_kern(*args)

    for key in ("linelum_gal", "linelum_bulge", "linelum_disk", "linelum_knots"):
        assert np.all(np.isfinite(getattr(dbk_specphot_info, key)))

    assert dbk_specphot_info.linelum_gal.shape == (n_gals, n_lines)
    assert dbk_specphot_info.linelum_bulge.shape == (n_gals, n_lines)
    assert dbk_specphot_info.linelum_disk.shape == (n_gals, n_lines)
    assert dbk_specphot_info.linelum_knots.shape == (n_gals, n_lines)

    component_lines_sum = (
        dbk_specphot_info.linelum_bulge
        + dbk_specphot_info.linelum_disk
        + dbk_specphot_info.linelum_knots
    )
    logdiff = np.log10(component_lines_sum) - np.log10(dbk_specphot_info.linelum_gal)
    assert np.allclose(logdiff, 0.0, atol=0.01)

    _ret3 = gd_dbk_kernels._get_dbk_phot_from_dbk_weights(
        dbk_specphot_info.ssp_photflux_table,
        dbk_weights,
        dbk_specphot_info.dust_frac_trans,
        dbk_specphot_info.frac_ssp_errors,
    )
    obs_mags_bulge, obs_mags_disk, obs_mags_knots = _ret3
    obs_mags_sum = -2.5 * np.log10(
        10 ** (-0.4 * obs_mags_bulge)
        + 10 ** (-0.4 * obs_mags_disk)
        + 10 ** (-0.4 * obs_mags_knots)
    )
    assert np.allclose(dbk_specphot_info.obs_mags, obs_mags_sum, atol=0.01)
