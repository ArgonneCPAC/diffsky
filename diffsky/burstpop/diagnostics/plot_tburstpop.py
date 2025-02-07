"""
"""

import numpy as np
from dsps.sfh import diffburst
from matplotlib import lines as mlines
from matplotlib import pyplot as plt

from .. import tburstpop as tbp


def make_tburstpop_comparison_plot(
    params,
    params2=tbp.DEFAULT_TBURSTPOP_PARAMS,
    fname=None,
    label1=r"${\rm new\ model}$",
    label2=r"${\rm default\ model}$",
):
    """Make basic diagnostic plot of the model for Tburst

    Parameters
    ----------
    params : namedtuple
        Instance of tburstpop.TburstPopParams

    params2 : namedtuple, optional
        Instance of tburstpop.TburstPopParams
        Default is set by DEFAULT_TBURSTPOP_PARAMS

    fname : string, optional
        filename of the output figure

    """
    lgyrarr = np.linspace(5, 9.05, 100)

    logsmarr = np.array((9.0, 9.0, 12.0, 12.0))
    logssfrarr = np.array((-7.0, -13.0, -7.0, -13.0))
    lgyr_peak, lgyr_max = tbp.get_tburst_params_from_tburstpop_params(
        tbp.DEFAULT_TBURSTPOP_PARAMS, logsmarr, logssfrarr
    )

    u_params = np.array(tbp.DEFAULT_TBURSTPOP_U_PARAMS) + np.random.uniform(
        -1, 1, len(tbp.DEFAULT_TBURSTPOP_PARAMS)
    )
    u_params = tbp.DEFAULT_TBURSTPOP_U_PARAMS._make(u_params)
    alt_params = tbp.get_bounded_tburstpop_params(u_params)
    lgyr_peak2, lgyr_max2 = tbp.get_tburst_params_from_tburstpop_params(
        alt_params, logsmarr, logssfrarr
    )

    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    ax.loglog()
    ax.set_xlim(8e4, 2e9)
    ax.set_ylim(2e-7, 10.5)
    xlabel = ax.set_xlabel(r"$\tau_{\rm age}\ {\rm [yr]}$")
    xlabel = ax1.set_xlabel(r"$\tau_{\rm age}\ {\rm [yr]}$")
    ylabel = ax.set_ylabel(r"${\rm PDF}$")

    age_weights = diffburst._pureburst_age_weights_from_params(
        lgyrarr, lgyr_peak[0], lgyr_max[0]
    )
    ax.plot(10**lgyrarr, age_weights, color="blue")
    age_weights2 = diffburst._pureburst_age_weights_from_params(
        lgyrarr, lgyr_peak2[0], lgyr_max2[0]
    )
    ax.plot(10**lgyrarr, age_weights2, "--", color="blue")

    age_weights = diffburst._pureburst_age_weights_from_params(
        lgyrarr, lgyr_peak[1], lgyr_max[1]
    )
    ax.plot(10**lgyrarr, age_weights, color="red")
    age_weights2 = diffburst._pureburst_age_weights_from_params(
        lgyrarr, lgyr_peak2[1], lgyr_max2[1]
    )
    ax.plot(10**lgyrarr, age_weights2, "--", color="red")

    age_weights = diffburst._pureburst_age_weights_from_params(
        lgyrarr, lgyr_peak[2], lgyr_max[2]
    )
    ax1.plot(10**lgyrarr, age_weights, color="blue")
    age_weights2 = diffburst._pureburst_age_weights_from_params(
        lgyrarr, lgyr_peak2[2], lgyr_max2[2]
    )
    ax1.plot(10**lgyrarr, age_weights2, "--", color="blue")

    age_weights = diffburst._pureburst_age_weights_from_params(
        lgyrarr, lgyr_peak[3], lgyr_max[3]
    )
    ax1.plot(10**lgyrarr, age_weights, color="red")
    age_weights2 = diffburst._pureburst_age_weights_from_params(
        lgyrarr, lgyr_peak2[3], lgyr_max2[3]
    )
    ax1.plot(10**lgyrarr, age_weights2, "--", color="red")

    ax.set_title(r"$M_{\star}=10^9M_{\odot}$")
    ax1.set_title(r"$M_{\star}=10^{12}M_{\odot}$")

    red_line = mlines.Line2D([], [], ls="-", c="red", label=r"${\rm Q}$")
    blue_line = mlines.Line2D([], [], ls="-", c="blue", label=r"${\rm SF}$")
    solid_line = mlines.Line2D(
        [], [], ls="-", c="gray", label=r"${\rm default\ model}$"
    )
    dashed_line = mlines.Line2D([], [], ls="--", c="gray", label=r"${\rm new\ model}$")
    leg0 = ax.legend(handles=[red_line, blue_line], loc="upper left")
    ax.add_artist(leg0)
    ax.legend(handles=[solid_line, dashed_line], loc="upper right")
    leg0 = ax1.legend(handles=[red_line, blue_line], loc="upper left")
    ax1.add_artist(leg0)
    ax1.legend(handles=[solid_line, dashed_line], loc="upper right")

    if fname is not None:
        fig.savefig(
            fname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
        )
    return fig
