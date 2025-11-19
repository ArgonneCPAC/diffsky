""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ...param_utils import diffsky_param_wrapper as dpw
from .. import mc_diffsky_disk_bulge_knot_seds as mc_dbk_sed
from . import test_lc_phot_kern as tlcphk


def test_disk_bulge_knot_phot_from_mock():
    ran_key = jran.key(0)

    lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing()
    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape
    assert lc_data.z_phot_table.shape == (n_z_table,)
    assert lc_data.ssp_data.ssp_lgmet.shape == (n_met,)
    assert lc_data.ssp_data.ssp_lg_age_gyr.shape == (n_age,)

    diffstarpop_params = dpw.DEFAULT_PARAM_COLLECTION[0]
    mzr_params = dpw.DEFAULT_PARAM_COLLECTION[1]
    spspop_params = dpw.DEFAULT_PARAM_COLLECTION[2]
    scatter_params = dpw.DEFAULT_PARAM_COLLECTION[3]
    ssp_err_pop_params = dpw.DEFAULT_PARAM_COLLECTION[4]

    mc_args = (
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.logmp0,
        lc_data.t_table,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        DEFAULT_COSMOLOGY,
    )
    dbk_phot_info = mc_dbk_sed._mc_diffsky_disk_bulge_knot_phot_kern(*mc_args)

    mock_args = (
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.logmp0,
        lc_data.t_table,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        DEFAULT_COSMOLOGY,
    )
    dbk_phot_info_from_mock = mc_dbk_sed._disk_bulge_knot_phot_from_mock(*mock_args)

    assert np.allclose(
        dbk_phot_info["obs_mags"], dbk_phot_info_from_mock["obs_mags"], rtol=1e-4
    )
