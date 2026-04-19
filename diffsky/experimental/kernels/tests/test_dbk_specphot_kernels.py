""""""

import numpy as np
import pytest
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ....param_utils import diffsky_param_wrapper as dpw
from ...tests import test_lightcone_generators as tlcg
from .. import dbk_specphot_kernels as dbkspk


def test_mc_dbk_specphot_kern(num_halos=250):
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


@pytest.mark.xfail
def test_mc_lc_dbk_sed_kern():
    raise NotImplementedError("Test not implemented yet")
