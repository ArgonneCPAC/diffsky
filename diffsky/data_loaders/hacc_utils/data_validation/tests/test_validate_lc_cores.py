""""""

import os

import numpy as np
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
def test_check_zrange_passes_for_example_last_journey_sky_patch():

    stepnum = 373
    lc_patch = 0

    bn_lc_cores = os.path.join(vlcc.BNPAT_LC_CORES.format(stepnum, lc_patch))
    fn_lc_cores = os.path.join(DRN_LC_CORES_POBOY, bn_lc_cores)
    msg = vlcc.check_zrange(fn_lc_cores, "LastJourney")

    assert len(msg) == 0
