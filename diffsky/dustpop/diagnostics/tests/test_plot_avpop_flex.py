"""
"""

import os

from ..plot_avpop_flex import DEFAULT_AVPOP_PARAMS, make_avpop_flex_comparison_plots

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def test_make_freqburst_comparison_plot():
    fn = os.path.join(_THIS_DRNAME, "dummy.png")
    _res = make_avpop_flex_comparison_plots(DEFAULT_AVPOP_PARAMS, fname=fn)
    (fig1, fnout1), (fig2, fnout2) = _res
    assert os.path.isfile(fnout1)
    os.remove(fnout1)
    assert os.path.isfile(fnout2)
    os.remove(fnout2)
