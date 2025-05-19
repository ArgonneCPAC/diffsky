"""
"""

import os

import numpy as np
import pytest
from dsps.data_loaders import retrieve_fake_fsps_data as rffd
from dsps.data_loaders.load_ssp_data import load_ssp_templates

from .. import phot_utils as pu

try:
    DSPS_DATA_DRN = os.environ["DSPS_DRN"]
    HAS_REAL_DSPS_DATA = True
except KeyError:
    DSPS_DATA_DRN = ""
    HAS_REAL_DSPS_DATA = False
HAS_DATA_MSG = "Must have DSPS_DRN in os.environ to run this test"


def test_load_fake_interpolated_lsst_tcurves():
    ssp_data = rffd.load_fake_ssp_data()
    lsst_tcurves_interp, lsst_tcurves_nointerp = pu.load_interpolated_lsst_curves(
        ssp_data.ssp_wave, drn_ssp_data=None
    )
    assert len(lsst_tcurves_interp) == 6
    for tcurve in lsst_tcurves_interp:
        assert np.all(np.isfinite(tcurve.wave))
        assert np.all(np.isfinite(tcurve.transmission))
        assert np.all(tcurve.transmission >= 0)
        assert np.all(tcurve.transmission <= 1)

    assert len(lsst_tcurves_nointerp) == 6
    for tcurve in lsst_tcurves_nointerp:
        assert np.all(np.isfinite(tcurve.wave))
        assert np.all(np.isfinite(tcurve.transmission))
        assert np.all(tcurve.transmission >= 0)
        assert np.all(tcurve.transmission <= 1)


@pytest.mark.skipif(not HAS_REAL_DSPS_DATA, reason=HAS_DATA_MSG)
def test_load_real_interpolated_lsst_tcurves():
    ssp_data = load_ssp_templates(drn=DSPS_DATA_DRN)
    lsst_tcurves_interp, lsst_tcurves_nointerp = pu.load_interpolated_lsst_curves(
        ssp_data.ssp_wave, drn_ssp_data=DSPS_DATA_DRN
    )
    for tcurves in (lsst_tcurves_interp, lsst_tcurves_nointerp):
        assert len(tcurves) == 6
        for tcurve in tcurves:
            assert np.all(np.isfinite(tcurve.wave))
            assert np.all(np.isfinite(tcurve.transmission))
            assert np.all(tcurve.transmission >= 0)
            assert np.all(tcurve.transmission <= 1)
