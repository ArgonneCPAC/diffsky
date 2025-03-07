"""
"""

import numpy as np
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from jax import random as jran

from .. import mc_diffsky as mcd

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


def plot_diffstar_frac_quenched(
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    diffstarpop_params2=DEFAULT_DIFFSTARPOP_PARAMS,
    fname="prob_quenched_vs_mstar_diffstarpop_cens.png",
    n_halos=100_000,
    ssfr_cut=-11.0,
):
    ran_key = jran.key(0)
    hosts_logmh_at_z = np.linspace(10, 15, n_halos)
    z_obs = 0.0
    diffstar_data = mcd.mc_diffstar_cenpop(
        ran_key,
        z_obs,
        hosts_logmh_at_z=hosts_logmh_at_z,
        diffstarpop_params=diffstarpop_params,
    )
    diffstar_data["logssfrh"] = np.log10(diffstar_data["sfh"]) - np.log10(
        diffstar_data["smh"]
    )
    n_t_table = diffstar_data["sfh"].shape[1]

    diffstar_data2 = mcd.mc_diffstar_cenpop(
        ran_key,
        z_obs,
        hosts_logmh_at_z=hosts_logmh_at_z,
        diffstarpop_params=diffstarpop_params2,
    )
    diffstar_data2["logssfrh"] = np.log10(diffstar_data2["sfh"]) - np.log10(
        diffstar_data2["smh"]
    )

    logsm_bins = np.linspace(8, 11.3, 30)
    logsm_binmids = 0.5 * (logsm_bins[:-1] + logsm_bins[1:])

    collector = []
    for it in range(n_t_table):
        x = np.log10(diffstar_data["smh"])[:, it]
        y = diffstar_data["logssfrh"][:, it] < ssfr_cut
        mean_fq_arr, __, __ = binned_statistic(x, y, bins=logsm_bins)
        collector.append(mean_fq_arr)
    mean_fq_cens = np.array(collector)

    collector = []
    for it in range(n_t_table):
        x = np.log10(diffstar_data2["smh"])[:, it]
        y = diffstar_data2["logssfrh"][:, it] < ssfr_cut
        mean_fq_arr, __, __ = binned_statistic(x, y, bins=logsm_bins)
        collector.append(mean_fq_arr)
    mean_fq_cens2 = np.array(collector)

    zarr = np.array((2.0, 1.0, 0.5, 0.0))
    _t_plot = flat_wcdm.age_at_z(zarr, *DEFAULT_COSMOLOGY)

    indx_t = np.array(
        [np.argmin(np.abs(diffstar_data["t_table"] - t)) for t in _t_plot]
    )
    t_plot = diffstar_data["t_table"][indx_t]
    _t_plot = np.array((2.0, 5.5, 9.0, 13.8))
    n_t_plot = t_plot.size
    colors = cm.coolwarm(np.linspace(1, 0, n_t_plot))  # red first

    labels = [r"$z=2$", r"$z=1$", r"$z=0.5$", r"$z=0$"]

    fig, ax = plt.subplots(1, 1)
    ax.set_xscale("log")
    xlim = ax.set_xlim(1e8, 3e11)
    ax.set_ylim(0.0, 1.5)
    xlabel = ax.set_xlabel(r"$M_{\star}\ [M_{\odot}]$")
    ylabel = ax.set_ylabel(r"$P({\rm sSFR}<-11)$")
    ax.set_title(r"${\rm DiffstarPop\ Quenching:\ Centrals}$")

    ax.plot(np.linspace(*xlim, 1000), np.ones(1000), ":", color="k")
    for it in range(n_t_plot - 1, -1, -1):
        ax.plot(10**logsm_binmids, mean_fq_cens[indx_t[it]], color=colors[it])
        ax.plot(10**logsm_binmids, mean_fq_cens2[indx_t[it]], "--", color=colors[it])

    leg_lines = [
        mlines.Line2D([], [], ls="-", c=colors[i], label=labels[i])
        for i in range(n_t_plot)
    ]
    leg1 = ax.legend(handles=leg_lines[::-1], loc="upper left")
    solid_line = mlines.Line2D([], [], ls="-", c="gray", label=r"${\rm new\ model}$")
    dashed_line = mlines.Line2D(
        [], [], ls="--", c="gray", label=r"${\rm default\ model}$"
    )
    ax.add_artist(leg1)
    ax.legend(handles=[solid_line, dashed_line], loc="upper right")

    fig.savefig(
        fname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )

    return fig
