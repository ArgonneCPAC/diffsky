""" """

import os

import numpy as np
import pytest
from dsps.data_loaders.defaults import (
    DEFAULT_SSP_BNAME_EMLINES as DEFAULT_SSP_BNAME_EMLINES_DSPS,
)

from .. import load_ssp_data
from ..defaults import DEFAULT_SSP_BNAME

DSPS_DATA_DRN = os.environ.get("DSPS_DRN", None)
ENV_VAR_MSG = "load_ssp_templates can only be tested if DSPS_DRN is in the env"


def test_default_diffsky_ssp_bname():
    """This unit test hard-codes the expectation for the default SSP data.
    This test should be updated whenever we change our default SSP choice."""
    assert DEFAULT_SSP_BNAME_EMLINES_DSPS == DEFAULT_SSP_BNAME


@pytest.mark.skipif(DSPS_DATA_DRN is None, reason=ENV_VAR_MSG)
def test_load_default_ssp_templates():
    ssp_data = load_ssp_data.load_ssp_templates(drn=DSPS_DATA_DRN, bn=DEFAULT_SSP_BNAME)
    assert ssp_data.ssp_emline_wave is not None
    assert ssp_data.ssp_emline_luminosity is not None

    for x in ssp_data:
        assert np.all(np.isfinite(x))

    # Enforce reasonable range of wavelength values for Angstrom units
    assert np.all(np.array(ssp_data.ssp_emline_wave) > 500)
    assert np.median(np.array(ssp_data.ssp_emline_wave)) > 1_000
    assert np.median(np.array(ssp_data.ssp_emline_wave)) < 10_000

    n_met = ssp_data.ssp_lgmet.size
    n_age = ssp_data.ssp_lg_age_gyr.size
    assert np.all(ssp_data.ssp_emline_luminosity.shape[:2] == (n_met, n_age))

    # Enforce reasonable range of luminosity values for erg/s/Msun units
    assert np.array(ssp_data.ssp_emline_luminosity).max() < 1e40
    assert np.array(ssp_data.ssp_emline_luminosity).min() > 0
    assert np.any(np.array(ssp_data.ssp_emline_luminosity) > 1e20)


@pytest.mark.skipif(DSPS_DATA_DRN is None, reason=ENV_VAR_MSG)
def test_write_ssp_templates_to_disk(tmp_path):
    """Verify that when we load the SSP data, write to disk,
    and then reload, we get the same result as the original"""
    ssp_data = load_ssp_data.load_ssp_templates(
        drn=DSPS_DATA_DRN, bn=load_ssp_data.DEFAULT_SSP_BNAME
    )
    fn_out = os.path.join(tmp_path, "dummy_ssp_data.hdf5")

    load_ssp_data.write_ssp_templates_to_disk(fn_out, ssp_data)
    ssp_data2 = load_ssp_data.load_ssp_templates(fn=fn_out)
    assert ssp_data2.ssp_emline_wave is not None

    assert set(ssp_data._fields) == set(ssp_data2._fields)
    for name in ssp_data._fields:
        arr = getattr(ssp_data, name)
        arr2 = getattr(ssp_data, name)
        assert np.allclose(arr, arr2, rtol=1e-4)


def test_load_fake_ssp_data():
    ssp_data = load_ssp_data.load_fake_ssp_data()
    assert ssp_data.ssp_emline_wave is not None

    assert ssp_data.ssp_emline_wave is not None
    assert ssp_data.ssp_emline_luminosity is not None

    for x in ssp_data:
        assert np.all(np.isfinite(x))

    # Enforce reasonable range of wavelength values for Angstrom units
    assert np.all(np.array(ssp_data.ssp_emline_wave) > 500)

    n_met = ssp_data.ssp_lgmet.size
    n_age = ssp_data.ssp_lg_age_gyr.size
    assert np.all(ssp_data.ssp_emline_luminosity.shape[:2] == (n_met, n_age))

    # Enforce reasonable range of luminosity values for erg/s/Msun units
    assert np.array(ssp_data.ssp_emline_luminosity).max() < 1e40
    assert np.array(ssp_data.ssp_emline_luminosity).min() > 0
    assert np.any(np.array(ssp_data.ssp_emline_luminosity) > 1e20)
