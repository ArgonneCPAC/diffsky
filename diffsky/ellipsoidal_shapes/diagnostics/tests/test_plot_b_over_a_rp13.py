""""""

from .. import plot_b_over_a_rp13 as pbsrp13


def test_make_bulge_rp13_comparison_plot():
    pbsrp13.make_bulge_rp13_comparison_plot(enforce_tol=0.2)


def test_make_disk_rp13_comparison_plot():
    pbsrp13.make_disk_rp13_comparison_plot(enforce_tol=0.2)
