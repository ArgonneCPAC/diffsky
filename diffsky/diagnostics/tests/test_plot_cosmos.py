""""""

import pytest

from .. import plot_cosmos as plc

ASTROPY_MSG = "Must have astropy installed to run this test"


@pytest.fixture(scope="module")
def testing_data(seed=0):
    """Generate lightcone data once per test module."""
    return _generate_testing_data(seed=seed)


def _generate_testing_data(seed=0):
    testing_data = plc.get_plotting_data(
        seed, num_halos=50, tcurves="random", ssp_data="random", cosmos="random"
    )
    return testing_data


@pytest.mark.skipif(not plc.HAS_ASTROPY, reason=ASTROPY_MSG)
def test_plot_app_mag_func(testing_data):
    plc.plot_app_mag_func(testing_data, 2.0, drn_out="FIGS")
