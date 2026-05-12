""""""

import pytest

from .. import plot_ssp_err_model as psspem


@pytest.mark.skipif(not psspem.HAS_MATPLOTLIB, reason=psspem.MATPLOTLIB_MSG)
def test_plot_ssp_err_model_delta_mag_vs_wavelength(tmp_path):
    psspem.plot_ssp_err_model_delta_mag_vs_wavelength(
        z_obs=0.0, drn_out=str(tmp_path), model_nickname="unit_testing_model"
    )
