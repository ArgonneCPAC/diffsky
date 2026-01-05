""" """

import pytest

from .. import plot_delta_mag_burstiness as pb


@pytest.mark.skipif(not pb.HAS_MATPLOTLIB, reason=pb.MATPLOTLIB_MSG)
def test_plot_delta_mag_lsst_vs_logsm():
    z_obs = 0.1
    pb.plot_delta_mag_lsst_vs_logsm(z_obs)


@pytest.mark.skipif(not pb.HAS_MATPLOTLIB, reason=pb.MATPLOTLIB_MSG)
def test_plot_delta_mag_lsst_vs_ssfr():
    z_obs = 0.1
    pb.plot_delta_mag_lsst_vs_ssfr(z_obs)
