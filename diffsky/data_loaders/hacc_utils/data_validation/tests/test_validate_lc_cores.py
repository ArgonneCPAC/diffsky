""""""

import os
from glob import glob

import pytest

from .. import validate_lc_cores as vlcc

DRN_LC_CORES_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc_cores"

try:
    from haccytrees import Simulation as HACCSim  # noqa

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False


try:
    assert os.path.isdir(DRN_LC_CORES_POBOY)
    CAN_RUN_LJ_DATA_TESTS = True
except AssertionError:
    CAN_RUN_LJ_DATA_TESTS = False
CAN_RUN_LJ_DATA_TESTS = CAN_RUN_LJ_DATA_TESTS & HAS_HACCYTREES
POBOY_MSG = "This test only runs on poboy machine with haccytrees installed"


@pytest.mark.skipif(not CAN_RUN_LJ_DATA_TESTS, reason=POBOY_MSG)
def test_check_zrange_passes_for_all_last_journey_data_on_poboy():
    """"""
    bnpat = vlcc.BNPAT_LC_CORES.format("*", "*")
    fn_list = glob(os.path.join(DRN_LC_CORES_POBOY, bnpat))
    for fn_lc_cores in fn_list:
        msg = vlcc.check_zrange(fn_lc_cores, "LastJourney")
        assert len(msg) == 0
