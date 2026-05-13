""""""

import numpy as np
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ....param_utils import diffsky_param_wrapper as dpw
from ...tests import test_mc_lightcone_halos as tmclh
from ...tests import test_mc_phot
from .. import gd_phot_kernels, phot_kernels


def test_mc_phot_kern():
    ran_key = jran.key(0)
    lc_data, tcurves = tmclh._get_weighted_lc_data_for_unit_testing()

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
    phot_kern_results, phot_randoms, diffstarpop_results = _res

    test_mc_phot.check_phot_kern_results(phot_kern_results)

    phot_kern_results2 = gd_phot_kernels._phot_kern(
        phot_randoms,
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

    for x, x2 in zip(phot_kern_results, phot_kern_results2):
        assert np.allclose(x, x2)

    phot_kern_results4, phot_randoms4 = phot_kernels._mc_phot_kern(
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
        [getattr(phot_kern_results4, x) for x in DEFAULT_DIFFSTAR_PARAMS._fields]
    )

    phot_kern_results3 = phot_kernels._phot_kern(
        phot_randoms4,
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

    return phot_kern_results, phot_kern_results2, phot_kern_results3, phot_kern_results4
    # assert np.allclose(
    #     phot_kern_results.obs_mags, phot_kern_results3.obs_mags, rtol=1e-5
    # )

    assert np.allclose(
        phot_kern_results3.obs_mags, phot_kern_results4.obs_mags, rtol=1e-5
    )
