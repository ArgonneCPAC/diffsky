""" """

import os

import numpy as np

from .. import ssp_err_model

try:
    import matplotlib.cm as cm
    from matplotlib import lines as mlines
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
MATPLOTLIB_MSG = "Must have matplotlib installed to use this function"

MBLUE = "#1f77b4"
MGREEN = "#2ca02c"
MRED = "#d62728"


def plot_ssp_err_model_delta_mag_vs_wavelength(
    *,
    ssp_err_params=ssp_err_model.DEFAULT_SSPERR_PARAMS,
    z_obs=0.0,
    drn_out="",
    model_nickname="default",
):
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG

    if drn_out != "":
        os.makedirs(drn_out, exist_ok=True)

    ngals = 500
    nwave = 5_000

    colors = cm.coolwarm(np.linspace(0, 1, ngals))  # blue first
    logsmarr = np.linspace(8, 12, ngals)
    wave_obs = np.linspace(1_000, 10_000, nwave)

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(-0.99, 0.99)
    xlim = ax.set_xlim(wave_obs.min(), wave_obs.max())
    ax.plot(np.linspace(*xlim, 100), np.zeros(100), ":", color="k")
    ax.fill_between(
        np.linspace(*xlim, 100),
        np.zeros(100) - 0.3,
        np.zeros(100) + 0.3,
        alpha=0.7,
        color="lightgray",
    )

    xlabel = ax.set_xlabel(r"$\lambda\ {\rm [\AA]}$")
    ylabel = ax.set_ylabel(r"$\delta{\rm L\ [mag]}$")

    for i in range(ngals):
        logsm = logsmarr[i]

        fracerr = ssp_err_model.frac_ssp_err_at_z_obs_singlegal(
            ssp_err_params, logsm, z_obs, wave_obs
        )

        ax.plot(wave_obs, -2.5 * np.log10(fracerr), color=colors[i])

    red_line = mlines.Line2D(
        [], [], ls="-", c=MRED, label=r"$M_{\star}=10^{12}M_{\odot}$"
    )
    blue_line = mlines.Line2D(
        [], [], ls="-", c=MBLUE, label=r"$M_{\star}=10^{8}M_{\odot}$"
    )
    ax.legend(handles=[red_line, blue_line])

    ax.set_title(rf"$z={z_obs:.1f}$")

    bnout = f"{model_nickname}_ssp_err_model_vs_wavelength_z={z_obs:.1f}.png"
    fnout = os.path.join(drn_out, bnout)

    fig.savefig(
        fnout, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )
    plt.close()
    return fig
