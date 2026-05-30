""""""

import numpy as np
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ....param_utils import diffsky_param_wrapper_merging as dpwm
from ... import dbk_phot_from_mock
from ... import mc_diffstarpop_wrappers as mcdpw
from ...tests import test_lightcone_generators as tlcg
from .. import gd_dbk_specphot_kernels as gd_dbkspk


def test_mc_dbk_phot_kern(num_halos=19):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.196
    upid = np.where(lc_data.is_central == 1, -1, lc_data.halo_indx)
    lgmu_infall = lc_data.logmp_infall - lc_data.logmhost_infall
    gyr_since_infall = lc_data.t_infall - lc_data.t_obs

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
        dpwm.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
        dpwm.DEFAULT_PARAM_COLLECTION.mzr_params,
        dpwm.DEFAULT_PARAM_COLLECTION.spspop_params,
        dpwm.DEFAULT_PARAM_COLLECTION.scatter_params,
        dpwm.DEFAULT_PARAM_COLLECTION.ssperr_params,
        dpwm.DEFAULT_PARAM_COLLECTION.merging_params,
        DEFAULT_COSMOLOGY,
        fb,
    )
    dbk_phot_info, dbk_weights = gd_dbkspk._mc_dbk_phot_kern(*args)


def test_mc_dbk_specphot_kern(num_halos=13):
    """Enforce that the sum of the component lines equals the composite line"""
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.196

    upid = np.where(lc_data.is_central == 1, -1, lc_data.halo_indx)
    lgmu_infall = lc_data.logmp_infall - lc_data.logmhost_infall
    gyr_since_infall = lc_data.t_infall - lc_data.t_obs

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
        dpwm.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
        dpwm.DEFAULT_PARAM_COLLECTION.mzr_params,
        dpwm.DEFAULT_PARAM_COLLECTION.spspop_params,
        dpwm.DEFAULT_PARAM_COLLECTION.scatter_params,
        dpwm.DEFAULT_PARAM_COLLECTION.ssperr_params,
        dpwm.DEFAULT_PARAM_COLLECTION.merging_params,
        DEFAULT_COSMOLOGY,
        fb,
    )
    dbk_specphot_info, dbk_weights = gd_dbkspk._mc_dbk_specphot_kern(*args)

    for key in ("linelum_gal", "linelum_bulge", "linelum_disk", "linelum_knots"):
        assert np.all(np.isfinite(getattr(dbk_specphot_info, key)))

    component_lines_sum = (
        dbk_specphot_info.linelum_bulge
        + dbk_specphot_info.linelum_disk
        + dbk_specphot_info.linelum_knots
    )
    logdiff = np.log10(component_lines_sum) - np.log10(dbk_specphot_info.linelum_gal)
    assert np.allclose(logdiff, 0.0, atol=0.01)


def test_dbk_sed_kern(num_halos=17):
    """Enforce that the sum of the component SEDs equals the composite SED"""
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.116
    upid = np.where(lc_data.is_central == 1, -1, lc_data.halo_indx)
    lgmu_infall = lc_data.logmp_infall - lc_data.logmhost_infall
    gyr_since_infall = lc_data.t_infall - lc_data.t_obs

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
    assert "diffstar_info_ms" in dbk_specphot_info._fields

    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [getattr(dbk_specphot_info, k) for k in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    diffstarpop_results = mcdpw.DiffstarPopResults(
        sfh_params=sfh_params,
        sfh_params_ms=dbk_specphot_info.diffstar_info_ms.sfh_params,
        sfh_params_q=dbk_specphot_info.diffstar_info_q.sfh_params,
        mc_is_q=dbk_specphot_info.mc_is_q,
        frac_q=dbk_specphot_info.frac_q,
    )

    dbk_sed_kern_args = (
        dbk_specphot_info.mc_is_q,
        dbk_specphot_info.uran_av,
        dbk_specphot_info.uran_delta,
        dbk_specphot_info.uran_funo,
        dbk_specphot_info.uran_pburst,
        dbk_specphot_info.delta_mag_ssp_scatter,
        dbk_specphot_info.uran_fbulge,
        dbk_specphot_info.fknot,
        diffstarpop_results,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        upid,
        lgmu_infall,
        lc_data.logmhost_infall,
        gyr_since_infall,
        lc_data.ssp_data,
        dpwm.DEFAULT_PARAM_COLLECTION.mzr_params,
        dpwm.DEFAULT_PARAM_COLLECTION.spspop_params,
        dpwm.DEFAULT_PARAM_COLLECTION.scatter_params,
        dpwm.DEFAULT_PARAM_COLLECTION.ssperr_params,
        dpwm.DEFAULT_PARAM_COLLECTION.merging_params,
        DEFAULT_COSMOLOGY,
        fb,
    )
    dbk_sed_info, __ = gd_dbkspk._dbk_sed_kern(*dbk_sed_kern_args)

    sed_sum = (
        dbk_sed_info.rest_sed_bulge
        + dbk_sed_info.rest_sed_disk
        + dbk_sed_info.rest_sed_knots
    )

    temp_args = (
        dbk_specphot_info.mc_is_q,
        dbk_specphot_info.uran_av,
        dbk_specphot_info.uran_delta,
        dbk_specphot_info.uran_funo,
        dbk_specphot_info.uran_pburst,
        dbk_specphot_info.delta_mag_ssp_scatter,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
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
    _res = dbk_phot_from_mock._reproduce_mock_sed_kern(*temp_args)
    phot_kern_results, phot_randoms, sed_kern_results = _res

    logdiff = np.log10(sed_sum) - np.log10(sed_kern_results.rest_sed)
    assert np.allclose(logdiff, 0.0, atol=0.01)
