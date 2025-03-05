"""
"""

import os

import numpy as np
from dsps.data_loaders import retrieve_fake_fsps_data as rffd

from .. import phot_utils as pu

try:
    DEFAULT_DSPS_DRN = os.environ["DSPS_DRN"]
except KeyError:
    DEFAULT_DSPS_DRN = ""


def test_get_interpolated_lsst_tcurves():
    ssp_data = rffd.load_fake_ssp_data()
    tcurves = pu.get_interpolated_lsst_tcurves(
        ssp_data.ssp_wave, drn_ssp_data=DEFAULT_DSPS_DRN
    )
    assert len(tcurves) == 6
    for tcurve in tcurves:
        assert np.all(np.isfinite(tcurve.wave))
        assert np.all(np.isfinite(tcurve.transmission))
        assert np.all(tcurve.transmission >= 0)
        assert np.all(tcurve.transmission <= 1)
