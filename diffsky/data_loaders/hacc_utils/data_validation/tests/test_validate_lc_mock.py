""""""

import os
from glob import glob

import pytest

from .. import validate_lc_mock as vlcm

DRN_LC_MOCK_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc_mock"

try:
    from haccytrees import Simulation as HACCSim  # noqa

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False


try:
    assert os.path.isdir(DRN_LC_MOCK_POBOY)
    CAN_RUN_LJ_DATA_TESTS = True
except AssertionError:
    CAN_RUN_LJ_DATA_TESTS = False
CAN_RUN_LJ_DATA_TESTS = CAN_RUN_LJ_DATA_TESTS & HAS_HACCYTREES
POBOY_MSG = "This test only runs on poboy machine with haccytrees installed"


@pytest.mark.skipif(not CAN_RUN_LJ_DATA_TESTS, reason=POBOY_MSG)
def test_check_all_columns_are_finite():
    """"""
    bnpat = vlcm.BNPAT_LC_MOCK.format("*", "*")
    fn_list = glob(os.path.join(DRN_LC_MOCK_POBOY, bnpat))
    for fn_lc_cores in fn_list:
        msg = vlcm.check_all_columns_are_finite(fn_lc_cores)
        assert len(msg) == 0
