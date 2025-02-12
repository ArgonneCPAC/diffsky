"""
"""

from .. import plot_delta_mag_burstiness as pb


def test_plot_delta_mag_lsst():
    z_obs = 0.1
    pb.plot_delta_mag_lsst(z_obs)
