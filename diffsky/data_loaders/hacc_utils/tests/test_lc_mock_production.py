""" """

import os

import pytest
from jax import random as jran

from .. import lc_mock_production as lcmp
from .. import load_lc_cf

DRN_CF_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"
DRN_LC_CF_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc-cf-diffsky"

HAS_HACCY_TREES = load_lc_cf.HAS_HACCYTREES

try:
    assert os.path.isdir(DRN_CF_LJ_POBOY)
    CAN_RUN_LJ_DATA_TESTS = True
except AssertionError:
    CAN_RUN_LJ_DATA_TESTS = False
CAN_RUN_LJ_DATA_TESTS = CAN_RUN_LJ_DATA_TESTS & HAS_HACCY_TREES
POBOY_MSG = "This test only runs on poboy machine with haccytrees installed"


@pytest.mark.skipif(not CAN_RUN_LJ_DATA_TESTS, reason=POBOY_MSG)
def test_add_sfh_quantities_to_mock():
    ran_key = jran.key(0)
    sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim("LastJourney")

    bn_list = ["lc_cores-213.0.diffsky_data.hdf5"]
    fn_list = [os.path.join(DRN_LC_CF_LJ_POBOY, bn) for bn in bn_list]
    lc_data, diffsky_data = load_lc_cf.collect_lc_diffsky_data(fn_list)

    args = (sim_info, lc_data, diffsky_data, ran_key)
    lc_data, diffsky_data = lcmp.add_sfh_quantities_to_mock(*args)
