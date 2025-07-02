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


@pytest.mark.skipif(not CAN_RUN_LJ_DATA_TESTS, reason=POBOY_MSG)
def test_check_core_tag_uniqueness():
    """Every core_tag should be unique. A very small number of repeats is allowed."""
    bnpat = vlcc.BNPAT_LC_CORES.format("*", "*")
    fn_list = glob(os.path.join(DRN_LC_CORES_POBOY, bnpat))
    for fn_lc_cores in fn_list:
        msg = vlcc.check_core_tag_uniqueness(fn_lc_cores)

        if len(msg) > 0:
            # No more than 10 core tags that have a repetition
            s = msg[1]
            n_distinct_repeats = int(s.split("=")[-1])
            assert n_distinct_repeats < 10

            # Only a single repetition is allowed
            s = msg[2]
            max_repetitions = int(s.split("=")[-1])
            assert max_repetitions == 2


@pytest.mark.skipif(not CAN_RUN_LJ_DATA_TESTS, reason=POBOY_MSG)
def test_check_top_host_tag_has_match():
    """top_host_tag should always agree with the result recalculated by diffsky

    In the current implementation, this test is skipped for lc_cores in which
    there is a repeated entry of core_tag

    """
    bnpat = vlcc.BNPAT_LC_CORES.format("*", "*")
    fn_list = glob(os.path.join(DRN_LC_CORES_POBOY, bnpat))
    for fn_lc_cores in fn_list:
        msg = vlcc.check_top_host_tag_has_match(fn_lc_cores)
        if len(msg) > 0:
            s = msg[0]
            if "Could not run test" in s:
                pass
            else:
                bn = os.path.basename(fn_lc_cores)
                raise ValueError(f"{bn} has mismatching top_host_idx")


@pytest.mark.skipif(not CAN_RUN_LJ_DATA_TESTS, reason=POBOY_MSG)
def test_check_top_host_idx_tag_agreement():
    """top_host_tag should always be consistent with top_host_idx"""
    bnpat = vlcc.BNPAT_LC_CORES.format("*", "*")
    fn_list = glob(os.path.join(DRN_LC_CORES_POBOY, bnpat))
    for fn_lc_cores in fn_list:
        msg = vlcc.check_top_host_idx_tag_agreement(fn_lc_cores)
        if len(msg) > 0:
            bn = os.path.basename(fn_lc_cores)
            raise ValueError(f"{bn} has mismatching top_host_idx")


@pytest.mark.skipif(not CAN_RUN_LJ_DATA_TESTS, reason=POBOY_MSG)
def test_check_host_pos_is_near_galaxy_pos():
    """"""
    bnpat = vlcc.BNPAT_LC_CORES.format("*", "*")
    fn_list = glob(os.path.join(DRN_LC_CORES_POBOY, bnpat))
    for fn_lc_cores in fn_list:
        msg = vlcc.check_host_pos_is_near_galaxy_pos(fn_lc_cores)
        assert len(msg) == 0, msg
