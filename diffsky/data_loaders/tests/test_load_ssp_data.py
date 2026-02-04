""" """

import os

import numpy as np
import pytest
from dsps.data_loaders.defaults import DEFAULT_SSP_BNAME as DEFAULT_DSPS_SSP_BNAME
from dsps.data_loaders.defaults import SSPData as DEFAULT_SSPData

from ..load_ssp_data import DEFAULT_DIFFSKY_SSP_BNAME, load_ssp_templates

DSPS_DATA_DRN = os.environ.get("DSPS_DRN", None)
ENV_VAR_MSG = "load_ssp_templates can only be tested if DSPS_DRN is in the env"


@pytest.mark.skipif(DSPS_DATA_DRN is None, reason=ENV_VAR_MSG)
def test_load_default_ssp_templates():
    ssp_data = load_ssp_templates(drn=DSPS_DATA_DRN, bn=DEFAULT_DSPS_SSP_BNAME)
    assert set(ssp_data._fields) == set(DEFAULT_SSPData._fields)


@pytest.mark.skipif(DSPS_DATA_DRN is None, reason=ENV_VAR_MSG)
def test_load_emline_ssp_templates():
    ssp_data = load_ssp_templates(drn=DSPS_DATA_DRN, bn=DEFAULT_DIFFSKY_SSP_BNAME)
    expected_keys = set(DEFAULT_SSPData._fields) | set(("emlines",))
    assert set(ssp_data._fields) == expected_keys

    n_met = ssp_data.ssp_lgmet.size
    n_age = ssp_data.ssp_lg_age_gyr.size

    for emline in ssp_data.emlines:
        assert np.isfinite(emline.line_wave)
        assert emline.line_wave > 500
        assert emline.line_wave < 50_000
        assert np.all(np.isfinite(emline.line_flux))
        assert np.all(emline.line_flux >= 0)
        assert np.any(emline.line_flux > 1e20)
        assert np.all(emline.line_flux < 1e40)
        assert emline.line_flux.shape == (n_met, n_age)
