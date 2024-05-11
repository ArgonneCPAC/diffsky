"""
"""

import numpy as np
from matplotlib import pyplot as plt

from ..funopop_simple import DEFAULT_FUNOPOP_PARAMS, get_funo_from_funopop_params


def make_funopop_comparison_plot(
    funopop_params,
    funopop_params2=DEFAULT_FUNOPOP_PARAMS,
    fname=None,
    label1=r"${\rm new\ model}$",
    label2=r"${\rm default\ model}$",
):
    """Make basic diagnostic plot of the model for the unobscured fraction, Funo

    Parameters
    ----------
    funopop_params : namedtuple
        Instance of funopop_simple.funopopParams

    funopop_params2 : namedtuple, optional
        Instance of funopop_simple.funopopParams
        Default is set by DEFAULT_funopop_PARAMS

    fname : string, optional
        filename of the output figure

    """
    nsm, nsfr = 250, 250
    logsm_grid = np.linspace(7, 12, nsm)
    logssfr_grid = np.linspace(-13, -8, nsfr)

    X, Y = np.meshgrid(logsm_grid, logssfr_grid)

    Z = get_funo_from_funopop_params(funopop_params, X, Y)
    Z2 = get_funo_from_funopop_params(funopop_params2, X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    (ax0, ax1) = axes

    pcm0 = ax0.pcolor(X, Y, Z, cmap="coolwarm_r", vmin=0, vmax=0.5)
    fig.colorbar(pcm0, ax=ax0)

    pcm1 = ax1.pcolor(X, Y, Z2, cmap="coolwarm_r", vmin=0, vmax=0.5)
    fig.colorbar(pcm1, ax=ax1, label=r"$F_{\rm unobs}$")
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
