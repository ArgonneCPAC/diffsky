""" """

import os

import numpy as np
import pytest
from dsps.data_loaders.defaults import DEFAULT_SSP_BNAME as DEFAULT_DSPS_SSP_BNAME
from dsps.data_loaders.defaults import SSPData as DEFAULT_SSPData

from .. import load_ssp_data

DSPS_DATA_DRN = os.environ.get("DSPS_DRN", None)
ENV_VAR_MSG = "load_ssp_templates can only be tested if DSPS_DRN is in the env"


@pytest.mark.skipif(DSPS_DATA_DRN is None, reason=ENV_VAR_MSG)
def test_load_default_ssp_templates():
    ssp_data = load_ssp_data.load_ssp_templates(
        drn=DSPS_DATA_DRN, bn=DEFAULT_DSPS_SSP_BNAME
    )
    assert set(ssp_data._fields) == set(DEFAULT_SSPData._fields)


@pytest.mark.skipif(DSPS_DATA_DRN is None, reason=ENV_VAR_MSG)
def test_load_emline_ssp_templates():
    ssp_data = load_ssp_data.load_ssp_templates(
        drn=DSPS_DATA_DRN, bn=load_ssp_data.DEFAULT_DIFFSKY_SSP_BNAME
    )
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


@pytest.mark.skipif(DSPS_DATA_DRN is None, reason=ENV_VAR_MSG)
def test_write_ssp_templates_to_disk(tmp_path):
    """Verify that when we load the SSP data, write to disk,
    and then reload, we get the same result as the original"""
    ssp_data = load_ssp_data.load_ssp_templates(
        drn=DSPS_DATA_DRN, bn=load_ssp_data.DEFAULT_DIFFSKY_SSP_BNAME
    )
    fn_out = os.path.join(tmp_path, "dummy_ssp_data.hdf5")

    load_ssp_data.write_ssp_templates_to_disk(fn_out, ssp_data)
    ssp_data2 = load_ssp_data.load_ssp_templates(fn=fn_out)

    assert set(ssp_data._fields) == set(ssp_data2._fields)
    for name in ssp_data._fields:
        if name != "emlines":
            arr = getattr(ssp_data, name)
            arr2 = getattr(ssp_data, name)
            assert np.allclose(arr, arr2, rtol=1e-4)

    for emline_name in ssp_data.emlines._fields:
        line = getattr(ssp_data.emlines, emline_name)
        line2 = getattr(ssp_data2.emlines, emline_name)
        assert np.allclose(line.line_wave, line2.line_wave, rtol=1e-3)
        assert np.allclose(line.line_flux, line2.line_flux, rtol=1e-3)


def test_load_fake_ssp_data():
    n_lines = 5
    ssp_data = load_ssp_data.load_fake_ssp_data(n_lines=n_lines)
    assert len(ssp_data.emlines) == 5

    emline_names = ["a", "b", "c"]
    ssp_data = load_ssp_data.load_fake_ssp_data(emline_names=emline_names)
    assert list(ssp_data.emlines._fields) == list(emline_names)
