""""""

import numpy as np
from matplotlib import cm
from matplotlib import lines as mlines
from matplotlib import pyplot as plt

from diffsky.mass_functions.hmf_model import predict_cuml_hmf

MRED = "#d62728"
MBLUE = "#1f77b4"


def make_hmf_fit_plot(loss_data_collector, p_best, figname="hmf_diagnostic.png"):

    colors = cm.coolwarm(np.linspace(1, 0, len(loss_data_collector)))  # red first

    fig, ax = plt.subplots(1, 1)
    ax.loglog()
    xlabel = ax.set_xlabel(r"$M_{\rm halo}\ {\rm [M_{\odot}]}$")
    ylabel = ax.set_ylabel(r"$n(>M_{\rm halo})\ (h/{\rm Mpc})^3$")

    for iz, loss_data_iz in enumerate(loss_data_collector):
        z, lgmp_bins, lgcuml_density = loss_data_iz
        ax.plot(10**lgmp_bins, 10**lgcuml_density, color=colors[iz])

        pred_lgcuml_density = predict_cuml_hmf(p_best, lgmp_bins, z)
        ax.plot(10**lgmp_bins, 10**pred_lgcuml_density, "--", color=colors[iz])

    z_lo = loss_data_collector[-1][0]
    z_hi = loss_data_collector[0][0]
    red_line = mlines.Line2D([], [], ls="-", c=MRED, label=f"z={z_hi:.1f}")
    blue_line = mlines.Line2D([], [], ls="-", c=MBLUE, label=f"z={z_lo:.1f}")
    leg1 = ax.legend(handles=[blue_line, red_line], loc="upper right", frameon=False)
    ax.add_artist(leg1)
    dashed_line = mlines.Line2D([], [], ls="--", c="k", label=r"${\rm HMF\ fit}$")
    solid_line = mlines.Line2D([], [], ls="-", c="k", label=r"${\rm target\ data}$")
    ax.legend(handles=[solid_line, dashed_line], loc="lower left", frameon=False)

    fig.savefig(
        figname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )
