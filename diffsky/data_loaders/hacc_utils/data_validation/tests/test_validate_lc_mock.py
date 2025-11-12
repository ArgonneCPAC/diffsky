""""""

import os
from glob import glob

from .. import validate_lc_mock as vlcm

DRN_LC_MOCK_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc_mock"


def test_check_all_columns_are_finite():
    """"""
    bnpat = vlcm.BNPAT_LC_MOCK.format("*", "*")
    fn_list = glob(os.path.join(DRN_LC_MOCK_POBOY, bnpat))
    for fn_lc_cores in fn_list:
        msg = vlcm.check_all_columns_are_finite(fn_lc_cores)
        assert len(msg) == 0
