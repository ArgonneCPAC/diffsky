"""
"""

import numpy as np
from matplotlib import pyplot as plt

from ..fburstpop import DEFAULT_FBURSTPOP_PARAMS, get_lgfburst_from_fburstpop_params


def make_fburstpop_comparison_plot(
    params,
    params2=DEFAULT_FBURSTPOP_PARAMS,
    fname=None,
    label1=r"${\rm new\ model}$",
    label2=r"${\rm default\ model}$",
):
    """Make basic diagnostic plot of the model for Fburst

    Parameters
    ----------
    params : namedtuple
        Instance of fburstpop.FburstPopParams

    params2 : namedtuple, optional
        Instance of fburstpop.FburstPopParams
        Default is set by DEFAULT_FBURSTPOP_PARAMS

    fname : string, optional
        filename of the output figure

    """
    nsm, nsfr = 250, 250
    logsm_grid = np.linspace(7, 12, nsm)
    logssfr_grid = np.linspace(-13, -8, nsfr)

    X, Y = np.meshgrid(logsm_grid, logssfr_grid)

    Z = get_lgfburst_from_fburstpop_params(params, X, Y)
    Z2 = get_lgfburst_from_fburstpop_params(params2, X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    (ax0, ax1) = axes

    pcm0 = ax0.pcolor(X, Y, Z, cmap="coolwarm_r", vmin=-4.5, vmax=-2.1)
    fig.colorbar(pcm0, ax=ax0)

    pcm1 = ax1.pcolor(X, Y, Z2, cmap="coolwarm_r", vmin=-4.5, vmax=-2.1)
    fig.colorbar(pcm1, ax=ax1, label=r"${\rm lgFburst}$")
    for ax in axes:
        xlabel = ax.set_xlabel(r"$\log_{10}M_{\star}/M_{\odot}$")
    ylabel = ax0.set_ylabel(r"${\rm \log_{10}sSFR}$")

    ax0.set_title(label1)
    ax1.set_title(label2)

    if fname is not None:
        fig.savefig(
            fname, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
        )
    return fig
