"""
"""

from .. import plot_delta_mag_burstiness as pb


def test_plot_delta_mag_lsst_vs_logsm():
    z_obs = 0.1
    pb.plot_delta_mag_lsst_vs_logsm(z_obs)


def test_plot_delta_mag_lsst_vs_ssfr():
    z_obs = 0.1
    pb.plot_delta_mag_lsst_vs_ssfr(z_obs)
