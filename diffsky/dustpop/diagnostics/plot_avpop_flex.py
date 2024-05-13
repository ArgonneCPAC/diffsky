"""
"""

import os

import numpy as np
from jax import jit as jjit
from jax import vmap
from matplotlib import pyplot as plt

from ..avpop_flex import DEFAULT_AVPOP_PARAMS, get_av_from_avpop_params_scalar

_A = (None, 0, None, None, None)
_B = (None, None, 0, None, None)
_C = (None, 0, 0, None, None)
get_av_kern = jjit(vmap(vmap(get_av_from_avpop_params_scalar, in_axes=_C), in_axes=_C))


def make_avpop_flex_comparison_plots(
    params,
    params2=DEFAULT_AVPOP_PARAMS,
    fname=None,
    zplot=(0.1, 2.0),
    lgageplot=[-4.0, 1.0],
):
    """Make basic diagnostic plot of the model for Fburst

    Parameters
    ----------
    params : namedtuple
        Instance of avpop_flex.AvPopParams

    params2 : namedtuple, optional
        Instance of avpop_flex.AvPopParams
        Default is set by DEFAULT_AVPOP_PARAMS

    fname : string, optional
        filename of the output figure

    """
    nsm, nsfr = 250, 250
    logsm_grid = np.linspace(7, 12, nsm)
    logssfr_grid = np.linspace(-13, -8, nsfr)

    X, Y = np.meshgrid(logsm_grid, logssfr_grid)

    ####################
    # Low-redshift plot

    redshift = zplot[0]
    fig1, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    (ax0, ax1), (ax2, ax3) = axes
    fig1.tight_layout(pad=2.0)

    for ax in (ax2, ax3):
        xlabel = ax.set_xlabel(r"$\log_{10}M_{\star}/M_{\odot}$")
    for ax in (ax0, ax2):
        ylabel = ax.set_ylabel(r"${\rm \log_{10}sSFR}$")

    title0 = ax0.set_title(f"new model: z={redshift:.2f}")
    ax1.set_title(f"default model: z={redshift:.2f}")

    Av_lgage0 = get_av_kern(params, X, Y, redshift, lgageplot[0])
    Av2_lgage0 = get_av_kern(params2, X, Y, redshift, lgageplot[0])
    Av_lgage1 = get_av_kern(params, X, Y, redshift, lgageplot[1])
    Av2_lgage1 = get_av_kern(params2, X, Y, redshift, lgageplot[1])

    ax0.pcolor(X, Y, Av_lgage0, cmap="coolwarm", vmin=0.0, vmax=2.0)

    pcm1 = ax1.pcolor(X, Y, Av2_lgage0, cmap="coolwarm", vmin=0.0, vmax=2.0)
    fig1.colorbar(pcm1, ax=ax1, label=r"${\rm Av}$")

    ax2.pcolor(X, Y, Av_lgage1, cmap="coolwarm", vmin=0.0, vmax=2.0)

    pcm3 = ax3.pcolor(X, Y, Av2_lgage1, cmap="coolwarm", vmin=0.0, vmax=2.0)
    fig1.colorbar(pcm3, ax=ax3, label=r"${\rm Av}$")

    row_label0 = r"${\rm young\ stars}$"
    row_label1 = r"${\rm old\ stars}$"
    ann1 = ax1.annotate(
        row_label0,
        xy=(11, -10.5),
        xytext=(14, -10.5),
        size=20,
        ha="right",
        va="center",
        rotation=-90,
    )

    ax3.annotate(
        row_label1,
        xy=(11, -10.5),
        xytext=(14.0, -10.5),
        size=20,
        ha="right",
        va="center",
        rotation=-90,
    )

    if fname is not None:
        bname = os.path.basename(fname)
        bname = f"redshift_{redshift:.2f}_" + bname
        fnout1 = os.path.join(os.path.dirname(fname), bname)
        fig1.savefig(
            fnout1,
            dpi=200,
            bbox_extra_artists=[title0, ann1, xlabel, ylabel],
            bbox_inches="tight",
        )

    ####################
    # High-redshift plot

    redshift = zplot[1]
    fig2, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    (ax0, ax1), (ax2, ax3) = axes
    fig2.tight_layout(pad=2.0)

    for ax in (ax2, ax3):
        xlabel = ax.set_xlabel(r"$\log_{10}M_{\star}/M_{\odot}$")
    for ax in (ax0, ax2):
        ylabel = ax.set_ylabel(r"${\rm \log_{10}sSFR}$")

    title_hiz = ax0.set_title(f"new model: z={redshift:.2f}")
    ax1.set_title(f"default model: z={redshift:.2f}")

    Av_lgage0 = get_av_kern(params, X, Y, redshift, lgageplot[0])
    Av2_lgage0 = get_av_kern(params2, X, Y, redshift, lgageplot[0])
    Av_lgage1 = get_av_kern(params, X, Y, redshift, lgageplot[1])
    Av2_lgage1 = get_av_kern(params2, X, Y, redshift, lgageplot[1])

    ax0.pcolor(X, Y, Av_lgage0, cmap="coolwarm", vmin=0.0, vmax=2.0)

    pcm1 = ax1.pcolor(X, Y, Av2_lgage0, cmap="coolwarm", vmin=0.0, vmax=2.0)
    fig2.colorbar(pcm1, ax=ax1, label=r"${\rm Av}$")

    ax2.pcolor(X, Y, Av_lgage1, cmap="coolwarm", vmin=0.0, vmax=2.0)

    pcm3 = ax3.pcolor(X, Y, Av2_lgage1, cmap="coolwarm", vmin=0.0, vmax=2.0)
    fig2.colorbar(pcm3, ax=ax3, label=r"${\rm Av}$")

    row_label0 = r"${\rm young\ stars}$"
    row_label1 = r"${\rm old\ stars}$"
    ann1_hiz = ax1.annotate(
        row_label0,
        xy=(11, -10.5),
        xytext=(14, -10.5),
        size=20,
        ha="right",
        va="center",
        rotation=-90,
    )

    ax3.annotate(
        row_label1,
        xy=(11, -10.5),
        xytext=(14.0, -10.5),
        size=20,
        ha="right",
        va="center",
        rotation=-90,
    )

    if fname is not None:
        bname = os.path.basename(fname)
        bname = f"redshift_{redshift:.2f}_" + bname
        fnout2 = os.path.join(os.path.dirname(fname), bname)
        fig2.savefig(
            fnout2,
            dpi=200,
            bbox_extra_artists=[title_hiz, ann1_hiz, xlabel, ylabel],
            bbox_inches="tight",
        )

    return (fig1, fnout1), (fig2, fnout2)
