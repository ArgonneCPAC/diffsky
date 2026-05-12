""" """

import os

import pytest

from .. import check_smhm

HAS_DEPENDENCIES = check_smhm.HAS_MATPLOTLIB & check_smhm.HAS_SCIPY
MSG_DEPS = "Must have matplotlib and scipy installed to run this test"


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason=MSG_DEPS)
def test_plot_diffstarpop_insitu_smhm(tmp_path):
    fname = os.path.join(tmp_path, "diffstarpop_insitu_smhm_lc_kern_check.png")
    check_smhm.plot_diffstarpop_insitu_smhm(fname=fname)
