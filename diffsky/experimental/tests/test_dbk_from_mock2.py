""""""

from collections import namedtuple

import numpy as np
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstar.defaults import FB
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.data_loaders.defaults import TransmissionCurve
from jax import random as jran

from ...param_utils import diffsky_param_wrapper as dpw
from .. import dbk_phot_from_mock
from .. import mc_lightcone_halos as mclh
from .. import mc_phot

SSP_DATA = retrieve_fake_fsps_data.load_fake_ssp_data()


ALL_FAKE_TCURVE_NAMES = ("u", "g", "r", "i", "z", "y")


def _get_weighted_lc_data_for_unit_testing(
    num_halos=50, ssp_data=SSP_DATA, tcurve_names=ALL_FAKE_TCURVE_NAMES
):
    ran_key = jran.key(0)

    lgmp_min, lgmp_max = 10.0, 15.0
    z_min, z_max = 0.1, 3.0
    sky_area_degsq = 100.0

    _res = retrieve_fake_fsps_data.load_fake_filter_transmission_curves()
    wave, u, g, r, i, z, y = _res

    tcurve_dict = dict(u=u, g=g, r=r, i=i, z=z, y=y)

    tcurves = []
    for name in tcurve_names:
        tcurves.append(TransmissionCurve(wave, tcurve_dict[name]))

    z_phot_table = 10 ** np.linspace(np.log10(z_min), np.log10(z_max), 30)

    args = (
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    lc_data = mclh.mc_weighted_lightcone_data(*args)

    return lc_data, tcurves


def test_reproduce_mock_dbk_kern():
    """Enforce that recomputed mock photometry agrees with original"""
    ran_key = jran.key(0)

    tcurve_names = ("u", "z")
    lc_data, tcurves = _get_weighted_lc_data_for_unit_testing(
        num_halos=50, tcurve_names=tcurve_names
    )
    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape
    assert lc_data.z_phot_table.shape == (n_z_table,)
    assert lc_data.ssp_data.ssp_lgmet.shape == (n_met,)
    assert lc_data.ssp_data.ssp_lg_age_gyr.shape == (n_age,)

    diffstarpop_params = dpw.DEFAULT_PARAM_COLLECTION[0]
    mzr_params = dpw.DEFAULT_PARAM_COLLECTION[1]
    spspop_params = dpw.DEFAULT_PARAM_COLLECTION[2]
    scatter_params = dpw.DEFAULT_PARAM_COLLECTION[3]
    ssp_err_pop_params = dpw.DEFAULT_PARAM_COLLECTION[4]

    dbk_phot_info = mc_phot.mc_lc_dbk_phot(
        ran_key,
        lc_data,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        DEFAULT_COSMOLOGY,
        FB,
    )
    dbk_phot_info = namedtuple("DBKPhotInfo", list(dbk_phot_info.keys()))(
        **dbk_phot_info
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
    _res = dbk_phot_from_mock._reproduce_mock_dbk_kern(*temp_args)
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


def test_reproduce_mock_phot_kern():
    """Enforce that recomputed mock photometry agrees with original"""
    ran_key = jran.key(0)

    tcurve_names = ("u", "z")
    lc_data, tcurves = _get_weighted_lc_data_for_unit_testing(
        num_halos=50, tcurve_names=tcurve_names
    )
    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape
    assert lc_data.z_phot_table.shape == (n_z_table,)
    assert lc_data.ssp_data.ssp_lgmet.shape == (n_met,)
    assert lc_data.ssp_data.ssp_lg_age_gyr.shape == (n_age,)

    diffstarpop_params = dpw.DEFAULT_PARAM_COLLECTION[0]
    mzr_params = dpw.DEFAULT_PARAM_COLLECTION[1]
    spspop_params = dpw.DEFAULT_PARAM_COLLECTION[2]
    scatter_params = dpw.DEFAULT_PARAM_COLLECTION[3]
    ssp_err_pop_params = dpw.DEFAULT_PARAM_COLLECTION[4]

    dbk_phot_info = mc_phot.mc_lc_dbk_phot(
        ran_key,
        lc_data,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        DEFAULT_COSMOLOGY,
        FB,
    )
    dbk_phot_info = namedtuple("DBKPhotInfo", list(dbk_phot_info.keys()))(
        **dbk_phot_info
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
    _res = dbk_phot_from_mock._reproduce_mock_phot_kern(*temp_args)
    phot_kern_results, phot_randoms = _res
    assert np.allclose(dbk_phot_info.obs_mags, phot_kern_results.obs_mags, rtol=1e-3)


def test_reproduce_mock_sed_kern():
    """Enforce that recomputed mock photometry agrees with original"""
    ran_key = jran.key(0)

    tcurve_names = ("u", "z")
    lc_data, tcurves = _get_weighted_lc_data_for_unit_testing(
        num_halos=50, tcurve_names=tcurve_names
    )
    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape
    assert lc_data.z_phot_table.shape == (n_z_table,)
    assert lc_data.ssp_data.ssp_lgmet.shape == (n_met,)
    assert lc_data.ssp_data.ssp_lg_age_gyr.shape == (n_age,)

    diffstarpop_params = dpw.DEFAULT_PARAM_COLLECTION[0]
    mzr_params = dpw.DEFAULT_PARAM_COLLECTION[1]
    spspop_params = dpw.DEFAULT_PARAM_COLLECTION[2]
    scatter_params = dpw.DEFAULT_PARAM_COLLECTION[3]
    ssp_err_pop_params = dpw.DEFAULT_PARAM_COLLECTION[4]

    dbk_phot_info = mc_phot.mc_lc_dbk_phot(
        ran_key,
        lc_data,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        DEFAULT_COSMOLOGY,
        FB,
    )
    dbk_phot_info = namedtuple("DBKPhotInfo", list(dbk_phot_info.keys()))(
        **dbk_phot_info
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
    _res = dbk_phot_from_mock._reproduce_mock_sed_kern(*temp_args)
    phot_kern_results, phot_randoms, sed_kern_results = _res
    assert np.allclose(dbk_phot_info.obs_mags, phot_kern_results.obs_mags, rtol=1e-3)

    rest_sed = sed_kern_results[0]
    assert np.all(np.isfinite(rest_sed))
