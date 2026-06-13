""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_emline_info as lemi
from jax import random as jran

from ....param_utils import diffsky_param_wrapper_merging as dpwm
from ...tests import test_lightcone_generators as tlcg
from .. import linelum_kernels, gd_phot_kernels


def test_mc_specphot_kern(num_halos=150):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.176
    ran_key, phot_key = jran.split(ran_key, 2)

    upid = np.where(lc_data.is_central == 1, -1, lc_data.halo_indx)
    lgmu_infall = lc_data.logmp_infall - lc_data.logmhost_infall
    gyr_since_infall = lc_data.t_infall - lc_data.t_obs
    _res = gd_phot_kernels._mc_phot_kern(
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
    phot_kern_results, phot_randoms, diffstarpop_results = _res

    n_lines = 3
    line_wave_table = np.linspace(1_000, 10_000, n_lines)
    emline_names = lc_data.ssp_data.ssp_emline_wave._fields[0:n_lines]
    ssp_data = lemi.get_subset_emline_data(lc_data.ssp_data, emline_names)
    lc_data = lc_data._replace(ssp_data=ssp_data)

    args = (
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
        line_wave_table,
        dpwm.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
        dpwm.DEFAULT_PARAM_COLLECTION.mzr_params,
        dpwm.DEFAULT_PARAM_COLLECTION.spspop_params,
        dpwm.DEFAULT_PARAM_COLLECTION.scatter_params,
        dpwm.DEFAULT_PARAM_COLLECTION.ssperr_params,
        dpwm.DEFAULT_PARAM_COLLECTION.merging_params,
        DEFAULT_COSMOLOGY,
        fb,
    )
    _specphot_res = linelum_kernels._mc_specphot_kern(*args)
    # _specphot_res = linelum_kernels._mc_specphot_kern(
    #     phot_key,
    #     lc_data.z_obs,
    #     lc_data.t_obs,
    #     lc_data.mah_params,
    #     lc_data.ssp_data,
    #     lc_data.precomputed_ssp_mag_table,
    #     lc_data.z_phot_table,
    #     lc_data.wave_eff_table,
    #     line_wave_table,
    #     dpwm.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
    #     dpwm.DEFAULT_PARAM_COLLECTION.mzr_params,
    #     dpwm.DEFAULT_PARAM_COLLECTION.spspop_params,
    #     dpwm.DEFAULT_PARAM_COLLECTION.scatter_params,
    #     dpwm.DEFAULT_PARAM_COLLECTION.ssperr_params,
    #     DEFAULT_COSMOLOGY,
    #     fb,
    # )

    phot_kern_results2, phot_randoms2, spec_kern_results = _specphot_res
    assert np.allclose(
        phot_kern_results.obs_mags, phot_kern_results2.obs_mags, rtol=1e-4
    )
