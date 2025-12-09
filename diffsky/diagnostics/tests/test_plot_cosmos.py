""""""

import pytest
from dsps.data_loaders import load_random_transmission_curve

from .. import plot_cosmos as plc


@pytest.fixture(scope="module")
def testing_data(seed=0):
    """Generate lightcone data once per test module."""
    return _generate_testing_data(seed=seed)


def _generate_testing_data(seed=0):
    testing_data = plc.get_plotting_data(seed, num_halos=50)
    return testing_data


def test_plot_app_mag_func(testing_data):
    plc.plot_app_mag_func(testing_data, 2.0, drn_out="FIGS")
