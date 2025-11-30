""""""

import numpy as np
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ...param_utils import diffsky_param_wrapper as dpw
from .. import dbk_from_mock2, mc_phot_repro
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

    dbk_phot_info = mc_phot_repro.mc_dbk_phot(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
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
        FB,
    )

    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [getattr(dbk_phot_info, pname) for pname in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    temp_args = (
        dbk_phot_info.mc_is_q,
        dbk_phot_info.uran_av,
        dbk_phot_info.uran_delta,
        dbk_phot_info.uran_funo,
        dbk_phot_info.uran_pburst,
        dbk_phot_info.delta_mag_ssp_scatter,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        dbk_phot_info.fknot,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        DEFAULT_COSMOLOGY,
        FB,
    )
    _res = dbk_from_mock2._reproduce_mock_phot_kern(*temp_args)
    phot_kern_results, phot_randoms, disk_bulge_history = _res[:3]
    obs_mags_bulge, obs_mags_disk, obs_mags_knots = _res[3:]
    assert np.allclose(dbk_phot_info.obs_mags, phot_kern_results.obs_mags, rtol=1e-3)

    assert np.allclose(dbk_phot_info.obs_mags_bulge, obs_mags_bulge, rtol=0.001)
    assert np.allclose(dbk_phot_info.obs_mags_disk, obs_mags_disk, rtol=0.001)
    assert np.allclose(dbk_phot_info.obs_mags_knots, obs_mags_knots, rtol=0.001)

    assert np.allclose(
        dbk_phot_info.bulge_to_total_history,
        disk_bulge_history.bulge_to_total_history,
        rtol=0.01,
    )
