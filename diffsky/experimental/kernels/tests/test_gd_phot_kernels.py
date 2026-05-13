""""""

import numpy as np
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ....param_utils import diffsky_param_wrapper as dpw
from ...tests import test_mc_lightcone_halos as tmclh
from ...tests import test_mc_phot
from .. import gd_phot_kernels, phot_kernels


def test_mc_phot_kern(num_halos=75):
    ran_key = jran.key(0)
    lc_data, tcurves = tmclh._get_weighted_lc_data_for_unit_testing(num_halos=num_halos)

    fb = 0.156
    ran_key, phot_key = jran.split(ran_key, 2)

    _res = gd_phot_kernels._mc_phot_kern(
        phot_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpw.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )
    mc_gd_phot_kern_results, mc_gd_phot_randoms, diffstarpop_results = _res

    test_mc_phot.check_phot_kern_results(mc_gd_phot_kern_results)

    return mc_gd_phot_kern_results

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

    gd_phot_kern_results = gd_phot_kernels._phot_kern(
        mc_gd_phot_randoms,
        diffstarpop_results,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
        dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
        dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
        dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
        DEFAULT_COSMOLOGY,
        fb,
    )

    for x, x2 in zip(mc_gd_phot_kern_results, gd_phot_kern_results):
        assert np.allclose(x, x2)

    mc_phot_kern_results, mc_phot_kern_randoms = phot_kernels._mc_phot_kern(
        phot_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        dpw.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
        dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
        dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
        dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
        dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
        DEFAULT_COSMOLOGY,
        fb,
    )

    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [getattr(mc_phot_kern_results, x) for x in DEFAULT_DIFFSTAR_PARAMS._fields]
    )

    phot_kern_results = phot_kernels._phot_kern(
        mc_phot_kern_randoms,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
        dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
        dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
        dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
        DEFAULT_COSMOLOGY,
        fb,
    )

    assert np.allclose(
        mc_gd_phot_kern_results.obs_mags, phot_kern_results.obs_mags, rtol=1e-5
    )

    assert np.allclose(
        phot_kern_results.obs_mags, mc_phot_kern_results.obs_mags, rtol=1e-5
    )
