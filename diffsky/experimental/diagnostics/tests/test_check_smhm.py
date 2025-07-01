""" """

import pytest

from .. import check_smhm

HAS_DEPENDENCIES = check_smhm.HAS_MATPLOTLIB & check_smhm.HAS_SCIPY
MSG_DEPS = "Must have matplotlib and scipy installed to run this test"


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason=MSG_DEPS)
def test_plot_diffstarpop_insitu_smhm():
    check_smhm.plot_diffstarpop_insitu_smhm()
