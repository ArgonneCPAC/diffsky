""""""

import pytest

from .. import plot_b_over_a_rp13 as pbsrp13


@pytest.mark.skipif(not pbsrp13.HAS_MATPLOTLIB, reason=pbsrp13.MATPLOTLIB_MSG)
def test_make_bulge_rp13_comparison_plot():
    pbsrp13.make_bulge_rp13_comparison_plot(enforce_tol=0.2)


@pytest.mark.skipif(not pbsrp13.HAS_MATPLOTLIB, reason=pbsrp13.MATPLOTLIB_MSG)
def test_make_disk_rp13_comparison_plot():
    pbsrp13.make_disk_rp13_comparison_plot(enforce_tol=0.2)
