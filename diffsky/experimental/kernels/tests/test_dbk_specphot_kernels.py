""""""

import numpy as np
import pytest
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ....param_utils import diffsky_param_wrapper as dpw
from ... import dbk_phot_from_mock
from ...tests import test_lightcone_generators as tlcg
from .. import dbk_specphot_kernels as dbkspk


def test_mc_dbk_specphot_kern(num_halos=13):
    """Enforce that the sum of the component lines equals the composite line"""
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.156

    args = (
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        lc_data.line_wave_table,
        *dpw.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )
    dbk_specphot_info, dbk_weights = dbkspk._mc_dbk_specphot_kern(*args)

    for key in ("linelum_gal", "linelum_bulge", "linelum_disk", "linelum_knots"):
        assert np.all(np.isfinite(getattr(dbk_specphot_info, key)))

    component_lines_sum = (
        dbk_specphot_info.linelum_bulge
        + dbk_specphot_info.linelum_disk
        + dbk_specphot_info.linelum_knots
    )
    logdiff = np.log10(component_lines_sum) - np.log10(dbk_specphot_info.linelum_gal)
    assert np.allclose(logdiff, 0.0, atol=0.01)


@pytest.mark.xfail
def test_mc_dbk_phot_kern():
    raise NotImplementedError("Test not implemented yet")


def test_mc_lc_dbk_sed_kern(num_halos=17):
    """Enforce that the sum of the component SEDs equals the composite SED"""
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.166

    args = (
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        lc_data.line_wave_table,
        *dpw.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )
    dbk_specphot_info, dbk_weights = dbkspk._mc_dbk_specphot_kern(*args)

    dbk_sed_info = dbkspk._mc_lc_dbk_sed_kern(
        dbk_specphot_info,
        dbk_weights,
        lc_data.z_obs,
        lc_data.ssp_data,
        dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
        dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
        dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
    )
    sed_sum = dbk_sed_info.sed_bulge + dbk_sed_info.sed_disk + dbk_sed_info.sed_knots

    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [getattr(dbk_specphot_info, pname) for pname in DEFAULT_DIFFSTAR_PARAMS._fields]
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
        dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
        dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
        dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
        dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
        DEFAULT_COSMOLOGY,
        fb,
    )
    _res = dbk_phot_from_mock._reproduce_mock_sed_kern(*temp_args)
    phot_kern_results, phot_randoms, sed_kern_results = _res

    logdiff = np.log10(sed_sum) - np.log10(sed_kern_results.rest_sed)
    assert np.allclose(logdiff, 0.0, atol=0.01)
