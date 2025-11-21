""""""

import numpy as np
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ...param_utils import diffsky_param_wrapper as dpw
from .. import dbk_from_mock
from .. import mc_diffsky_disk_bulge_knot_seds as mc_dbk_sed
from . import test_lc_phot_kern as tlcphk


def test_disk_bulge_knot_phot_from_mock():
    """Enforce that recomputed mock photometry agrees with original"""
    ran_key = jran.key(0)

    lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing(num_halos=500)
    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape
    assert lc_data.z_phot_table.shape == (n_z_table,)
    assert lc_data.ssp_data.ssp_lgmet.shape == (n_met,)
    assert lc_data.ssp_data.ssp_lg_age_gyr.shape == (n_age,)

    diffstarpop_params = dpw.DEFAULT_PARAM_COLLECTION[0]
    mzr_params = dpw.DEFAULT_PARAM_COLLECTION[1]
    spspop_params = dpw.DEFAULT_PARAM_COLLECTION[2]
    scatter_params = dpw.DEFAULT_PARAM_COLLECTION[3]
    ssp_err_pop_params = dpw.DEFAULT_PARAM_COLLECTION[4]

    n_gals = lc_data.z_obs.size
    fb = 0.156
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
        fb,
    )

    dbk_phot_info = mc_dbk_sed._mc_diffsky_disk_bulge_knot_phot_kern(*mc_args)

    mock_args = (
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

    _msk_q = dbk_phot_info["mc_sfh_type"].reshape((n_gals, 1))
    delta_scatter = np.where(
        _msk_q == 0, dbk_phot_info["delta_scatter_q"], dbk_phot_info["delta_scatter_ms"]
    )

    diffstar_params = [dbk_phot_info[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    diffstar_params = DEFAULT_DIFFSTAR_PARAMS._make(diffstar_params)

    fb = 0.156

    mock_args = (
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.logmp0,
        lc_data.t_table,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        diffstar_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        DEFAULT_COSMOLOGY,
        fb,
        dbk_phot_info["uran_av"],
        dbk_phot_info["uran_delta"],
        dbk_phot_info["uran_funo"],
        delta_scatter,
        dbk_phot_info["mc_sfh_type"],
        dbk_phot_info["fknot"],
    )

    dbk_phot_info_from_mock = dbk_from_mock._disk_bulge_knot_phot_from_mock(*mock_args)

    assert np.allclose(
        dbk_phot_info["obs_mags"], dbk_phot_info_from_mock["obs_mags"], rtol=1e-3
    )

    assert np.allclose(
        dbk_phot_info["obs_mags_bulge"],
        dbk_phot_info_from_mock["obs_mags_bulge"],
        rtol=0.001,
    )

    assert np.allclose(
        dbk_phot_info["obs_mags_disk"],
        dbk_phot_info_from_mock["obs_mags_disk"],
        rtol=0.01,
    )

    assert np.allclose(
        dbk_phot_info["bulge_to_total_history"],
        dbk_phot_info_from_mock["bulge_to_total_history"],
        rtol=0.01,
    )

    assert np.allclose(
        dbk_phot_info["obs_mags_knots"],
        dbk_phot_info_from_mock["obs_mags_knots"],
        atol=0.1,
    )
