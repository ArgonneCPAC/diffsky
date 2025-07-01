""" """

import numpy as np
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from jax import random as jran

from .. import mc_lightcone_halos as mclh

try:
    from matplotlib import cm
    from matplotlib import lines as mlines
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy.stats import binned_statistic

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

MRED = "#d62728"
MBLUE = "#1f77b4"


def plot_diffstarpop_insitu_smhm(
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    diffstarpop_params2=DEFAULT_DIFFSTARPOP_PARAMS,
    fname="diffstarpop_insitu_smhm_lc_kern_check.png",
):
    ran_key = jran.key(0)
    lgmp_min = 11.0
    sky_area_degsq = 20.0
    sky_area_degsq_list = [100.0, 20.0, 1.0, 1.0]

    logmp_bins = np.linspace(lgmp_min, 14, 30)
    logmp_bin_mids = 0.5 * (logmp_bins[:-1] + logmp_bins[1:])

    z_list = (0.05, 0.5, 1.0, 2.0)
    collector = []
    for z, sky_area_degsq in zip(z_list, sky_area_degsq_list):
        ran_key, z_key = jran.split(ran_key, 2)
        z_min, z_max = z - 0.01, z + 0.01
        z_min, z_max = z - 0.01, z + 0.01

        args = (z_key, lgmp_min, z_min, z_max, sky_area_degsq)
        cenpop = mclh.mc_lightcone_diffstar_cens(
            *args, diffstarpop_params=diffstarpop_params
        )
        cenpop2 = mclh.mc_lightcone_diffstar_cens(
            *args, diffstarpop_params=diffstarpop_params2
        )
        mean_logsm, __, __ = binned_statistic(
            cenpop["logmp_obs"], cenpop["logsm_obs"], bins=logmp_bins, statistic="mean"
        )
        mean_logsm2, __, __ = binned_statistic(
            cenpop2["logmp_obs"],
            cenpop2["logsm_obs"],
            bins=logmp_bins,
            statistic="mean",
        )
        collector.append((mean_logsm, mean_logsm2))

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(8, 12)
    xlabel = ax.set_xlabel(r"${\log_{10}M_{\rm h}}$")
    ylabel = ax.set_ylabel(r"${\log_{10}M_{\star}}$")

    colors = cm.coolwarm(np.linspace(0, 1, len(z_list)))  # blue first

    for i, plot_data in enumerate(collector):
        mean_logsm, mean_logsm2 = plot_data

        ax.plot(logmp_bin_mids, mean_logsm, color=colors[i])
        ax.plot(logmp_bin_mids, mean_logsm2, "--", color=colors[i])

    red_line = mlines.Line2D([], [], ls="-", c=MRED, label=r"$z=2$")
    blue_line = mlines.Line2D([], [], ls="-", c=MBLUE, label=r"$z=0$")
    leg1 = ax.legend(handles=[red_line, blue_line], loc="upper left")
    ax.add_artist(leg1)

    dashed_line = mlines.Line2D([], [], ls="-", c="gray", label=r"model 1")
    solid_line = mlines.Line2D([], [], ls="--", c="gray", label=r"model 2")
    ax.legend(handles=[dashed_line, solid_line], loc="lower right")

    fig.savefig(
        fname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )

    return fig
