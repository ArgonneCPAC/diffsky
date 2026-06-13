""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ....param_utils import diffsky_param_wrapper_merging as dpwm
from ...tests import test_lightcone_generators as tlcg
from .. import dbk_specphot_kernels as gd_dbkspk


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
