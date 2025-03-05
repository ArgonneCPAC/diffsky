"""
"""

import pytest

from .. import plot_diffstar_fq as pdq

HAS_DEPENDENCIES = pdq.HAS_MATPLOTLIB & pdq.HAS_SCIPY
MSG_DEPS = "Must have matplotlib and scipy installed to run this test"


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason=MSG_DEPS)
def test_plot_diffstar_frac_quenched():
    pdq.plot_diffstar_frac_quenched(n_halos=2_000)
