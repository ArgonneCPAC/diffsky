""" """

import os

import pytest

from ..plot_pmerge_vs_time import (
    HAS_MATPLOTLIB,
    MATPLOTLIB_MSG,
    make_pmerge_vs_time_plot,
)

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason=MATPLOTLIB_MSG)
def test_make_pmerge_vs_time_plot():
    fn = os.path.join(_THIS_DRNAME, "dummy.png")
    make_pmerge_vs_time_plot(fname=fn)
    assert os.path.isfile(fn)
    os.remove(fn)
