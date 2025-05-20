""" """

import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt

from .. import merging_model as mmod

mred = "#d62728"
mblue = "#1f77b4"


def make_pmerge_vs_time_plot(params=mmod.DEFAULT_MERGE_PARAMS, fname=None):

    ncolors = 200
    colors = cm.coolwarm(np.linspace(1, 0, ncolors))  # red first

    T_OBS = 13.8
    n = 10_000
    t_infall = np.linspace(1, T_OBS, n)
    t_since_infall = T_OBS - t_infall

    fig, _axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    fig.tight_layout(pad=4.0)
    ((ax0, ax1), (ax2, ax3)) = _axes
    axes = ax0, ax1, ax2, ax3
    ax0.set_ylim(-0.02, 1.02)

    titles = list(
        (
            r"$M_{\rm host}=10^{12}M_{\odot}$",
            r"$M_{\rm host}=10^{13}M_{\odot}$",
            r"$M_{\rm host}=10^{14}M_{\odot}$",
            r"$M_{\rm host}=10^{15}M_{\odot}$",
        )
    )

    for iax, ax in enumerate(axes):
        logmhost = np.array((12, 13, 14, 15))[iax]
        ax.set_title(titles[iax])
        for i in range(ncolors):
            logmp = np.linspace(10, logmhost, ncolors)[i]
            args = (params, logmp, logmhost, T_OBS, t_infall, 0)
            p_merge = mmod.get_p_merge_from_merging_params(*args)

            ax.plot(t_since_infall, p_merge, color=colors[i])

    for ax in axes:
        xlabel = ax.set_xlabel(r"${\rm time\ since\ infall\ [Gyr]}$")
        ylabel = ax.set_ylabel(r"$P_{\rm merge}$")

    from matplotlib import lines as mlines

    red_line = mlines.Line2D(
        [], [], ls="-", c=mred, label=r"$M_{\rm sub}=10^{10}M_{\odot}$"
    )
    blue_line = mlines.Line2D(
        [], [], ls="-", c=mblue, label=r"$M_{\rm sub}=M_{\rm host}$"
    )
    ax0.legend(handles=[red_line, blue_line])

    if fname is not None:
        fig.savefig(
            fname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
        )
    return fig
