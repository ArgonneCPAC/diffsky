""" """

import os

import numpy as np
import pytest

from .. import load_lc_cf

DRN_CF_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"
DRN_LC_CF_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc-cf-diffsky"

HAS_HACCY_TREES = load_lc_cf.HAS_HACCYTREES
NO_HACCY_TREES_MSG = "Must have haccytrees installed to run this test"

try:
    assert os.path.isdir(DRN_CF_LJ_POBOY)
    CAN_RUN_LJ_DATA_TESTS = True
except AssertionError:
    CAN_RUN_LJ_DATA_TESTS = False
CAN_RUN_LJ_DATA_TESTS = CAN_RUN_LJ_DATA_TESTS & HAS_HACCY_TREES
POBOY_MSG = "This test only runs on poboy machine with haccytrees installed"


@pytest.mark.skipif(not CAN_RUN_LJ_DATA_TESTS, reason=POBOY_MSG)
def test_get_diffsky_info_from_hacc_sim():
    sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim("LastJourney")

    assert np.allclose(0.158164, sim_info.fb, rtol=1e-4)


@pytest.mark.skipif(not CAN_RUN_LJ_DATA_TESTS, reason=POBOY_MSG)
def test_collect_lc_diffsky_data():
    bn_list = ["lc_cores-131.0.diffsky_data.hdf5", "lc_cores-213.0.diffsky_data.hdf5"]
    fn_list = [os.path.join(DRN_LC_CF_LJ_POBOY, bn) for bn in bn_list]
    lc_data, diffsky_data = load_lc_cf.collect_lc_diffsky_data(fn_list)
    assert len(lc_data["redshift_true"]) == len(diffsky_data["early_index"])
