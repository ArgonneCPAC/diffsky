"""
"""

import os

from ..plot_fburstpop import DEFAULT_FBURSTPOP_PARAMS, make_fburstpop_comparison_plot

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def test_make_freqburst_comparison_plot():
    fn = os.path.join(_THIS_DRNAME, "dummy.png")
    make_fburstpop_comparison_plot(DEFAULT_FBURSTPOP_PARAMS, fname=fn)
    assert os.path.isfile(fn)
    os.remove(fn)
