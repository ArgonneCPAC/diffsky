""" """

import os

import numpy as np
import pytest

from .. import load_lc_cf

DRN_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"

NO_HACC_MSG = "Must have haccytrees installed to run this test"
POBOY_MSG = "This test only runs on poboy machine"

try:
    assert os.path.isdir(DRN_LJ_POBOY)
    assert load_lc_cf.HAS_HACCYTREES
    CAN_RUN_HACC_DATA_TESTS = True
except AssertionError:
    CAN_RUN_HACC_DATA_TESTS = False


@pytest.mark.skipif(not CAN_RUN_HACC_DATA_TESTS, reason=POBOY_MSG)
def test_get_diffsky_info_from_hacc_sim():
    sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim("LastJourney")

    assert np.allclose(0.158164, sim_info.fb, rtol=1e-4)
