""""""

import pytest

from ...param_utils import diffsky_param_wrapper as dpw
from .. import plot_cosmos as plc

ASTROPY_MSG = "Must have astropy installed to run this test"


@pytest.fixture(scope="module")
def testing_data(seed=0):
    """Generate lightcone data once per test module."""
    return _generate_testing_data(seed=seed)


def _generate_testing_data(seed=0):
    testing_data = plc.get_plotting_data(
        seed=seed, num_halos=50, tcurves="random", ssp_data="random", cosmos="random"
    )
    return testing_data


@pytest.mark.skipif(not plc.HAS_MATPLOTLIB, reason=plc.MATPLOTLIB_MSG)
@pytest.mark.skipif(not plc.HAS_ASTROPY, reason=ASTROPY_MSG)
def test_plot_app_mag_func(testing_data, tmp_path):
    plc.plot_app_mag_func(pdata=testing_data, z_bin=2.0, drn_out=str(tmp_path))


@pytest.mark.skipif(not plc.HAS_MATPLOTLIB, reason=plc.MATPLOTLIB_MSG)
@pytest.mark.skipif(not plc.HAS_ASTROPY, reason=ASTROPY_MSG)
def test_plot_color_pdf(testing_data, tmp_path):
    plc.plot_color_pdf(
        pdata=testing_data,
        m1_bin=23.0,
        c0="UVISTA_Y_MAG",
        c1="UVISTA_J_MAG",
        z_bin=1.5,
        drn_out=str(tmp_path),
    )


@pytest.mark.skipif(not plc.HAS_MATPLOTLIB, reason=plc.MATPLOTLIB_MSG)
@pytest.mark.skipif(not plc.HAS_ASTROPY, reason=ASTROPY_MSG)
def test_make_color_mag_diagnostic_plots(testing_data, tmp_path):
    plc.make_color_mag_diagnostic_plots(
        pdata=testing_data,
        param_collection=dpw.DEFAULT_PARAM_COLLECTION,
        model_nickname="default",
        drn_out=str(tmp_path),
    )
