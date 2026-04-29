""""""

import os
import numpy as np
from glob import glob
from .. import load_lc_mock as llcm

from scipy.stats import binned_statistic

try:
    from matplotlib import pyplot as plt
    from matplotlib import lines as mlines

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
MATPLOTLIB_MSG = "Must have matplotlib installed to use this function"

try:
    from astropy.cosmology import FlatLambdaCDM

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
ASTROPY_MSG = "Must have astropy installed to use this function"

Z_MIN, Z_MAX = 0.0, 2.0

MBLUE = "#1f77b4"
MGREEN = "#2ca02c"
MORANGE = "#ff7f0e"
MRED = "#d62728"

HOD_HALOPROPS = [
    "redshift_true",
    "central",
    "logsm_obs",
    "logmp_obs",
    "logmp_obs_host",
    "logssfr_obs",
]


def get_plotting_data_mock(
    *,
    patch_list,
    drn_mock,
    drn_synthetic_cores=None,
    z_min=Z_MIN,
    z_max=Z_MAX,
    sim_name="LastJourney",
    keys=HOD_HALOPROPS,
):
    """"""
    pdata = llcm.load_diffsky_lightcone(
        drn_mock, sim_name, z_min, z_max, patch_list, keys=keys
    )
    fn_lc_mock = glob(os.path.join(drn_mock, "lc_cores*.hdf5"))[0]
    metadata = llcm.load_mock_metadata(fn_lc_mock)

    return pdata, metadata


def plot_smf(*, pdata, metadata, z_bin, dz=0.25, drn_out=""):
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG
    assert HAS_ASTROPY, ASTROPY_MSG

    drn_out = drn_out or "."
    os.makedirs(drn_out, exist_ok=True)

    dsps_cosmo_mock = metadata["sim_info"].cosmo_params
    astropy_cosmo = FlatLambdaCDM(H0=100 * dsps_cosmo_mock.h, Om0=dsps_cosmo_mock.Om0)
    area_subvol = 31.0  # deg

    logsm_bins = np.linspace(9, 12, 50)
    dlogsm = np.diff(logsm_bins).mean()

    fig, ax = plt.subplots(1, 1)
    __ = ax.loglog()
    xlabel = ax.set_xlabel(r"$M_{\star}\ [M_{\odot}]$")
    ylabel = ax.set_ylabel(r"$\Phi\ {\rm [1/Mpc^{3}/dex]}$")
    ax.set_title(r"${\rm Stellar\ Mass\ Function}$")

    zlo, zhi = z_bin[0] - dz / 2, z_bin[0] + dz / 2
    msk_z = (pdata["redshift_true"] > zlo) & (pdata["redshift_true"] < zhi)
    y, __ = np.histogram(pdata["logsm_obs"][msk_z], bins=logsm_bins)
    vol_4pi = astropy_cosmo.comoving_volume(zhi) - astropy_cosmo.comoving_volume(zlo)
    vol_subvol = (area_subvol / 40_000) * vol_4pi
    __ = ax.plot(
        10 ** logsm_bins[1:],
        y / dlogsm / vol_subvol,
        drawstyle="steps",
        color=MBLUE,
        label=r"${z=0.3}$",
    )

    zlo, zhi = z_bin[1] - dz / 2, z_bin[1] + dz / 2
    msk_z = (pdata["redshift_true"] > zlo) & (pdata["redshift_true"] < zhi)
    y, __ = np.histogram(pdata["logsm_obs"][msk_z], bins=logsm_bins)
    vol_4pi = astropy_cosmo.comoving_volume(zhi) - astropy_cosmo.comoving_volume(zlo)
    vol_subvol = (area_subvol / 40_000) * vol_4pi
    __ = ax.plot(
        10 ** logsm_bins[1:],
        y / dlogsm / vol_subvol,
        drawstyle="steps",
        color=MGREEN,
        label=r"${z=0.9}$",
    )

    zlo, zhi = z_bin[2] - dz / 2, z_bin[2] + dz / 2
    msk_z = (pdata["redshift_true"] > zlo) & (pdata["redshift_true"] < zhi)
    y, __ = np.histogram(pdata["logsm_obs"][msk_z], bins=logsm_bins)
    vol_4pi = astropy_cosmo.comoving_volume(zhi) - astropy_cosmo.comoving_volume(zlo)
    vol_subvol = (area_subvol / 40_000) * vol_4pi
    __ = ax.plot(
        10 ** logsm_bins[1:],
        y / dlogsm / vol_subvol,
        drawstyle="steps",
        color=MORANGE,
        label=r"${z=1.3}$",
    )

    zlo, zhi = z_bin[3] - dz / 2, z_bin[3] + dz / 2
    msk_z = (pdata["redshift_true"] > zlo) & (pdata["redshift_true"] < zhi)
    y, __ = np.histogram(pdata["logsm_obs"][msk_z], bins=logsm_bins)
    vol_4pi = astropy_cosmo.comoving_volume(zhi) - astropy_cosmo.comoving_volume(zlo)
    vol_subvol = (area_subvol / 40_000) * vol_4pi
    ax.plot(
        10 ** logsm_bins[1:],
        y / dlogsm / vol_subvol,
        drawstyle="steps",
        color=MRED,
        label=r"${z=1.9}$",
    )

    ax.legend()
    fn_out = os.path.join(drn_out, "smf_analysis_cosmos_260316_04_26_2026.png")
    fig.savefig(
        fn_out, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )

    plt.close()
    return fig


def plot_fsat(*, pdata, z_bin, dz=0.5, drn_out=""):
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG
    assert HAS_ASTROPY, ASTROPY_MSG

    drn_out = drn_out or "."
    os.makedirs(drn_out, exist_ok=True)

    logsm_bins = np.linspace(9, 12, 50)

    msk_z01 = np.abs(pdata["redshift_true"] - z_bin[0]) < dz
    mean_fsat01, __, __ = binned_statistic(
        pdata["logsm_obs"][msk_z01], pdata["central"][msk_z01] == 0, bins=logsm_bins
    )

    msk_z12 = np.abs(pdata["redshift_true"] - z_bin[1]) < dz
    mean_fsat12, __, __ = binned_statistic(
        pdata["logsm_obs"][msk_z12], pdata["central"][msk_z12] == 0, bins=logsm_bins
    )

    msk_z23 = np.abs(pdata["redshift_true"] - z_bin[2]) < dz
    mean_fsat23, __, __ = binned_statistic(
        pdata["logsm_obs"][msk_z23], pdata["central"][msk_z23] == 0, bins=logsm_bins
    )

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(0, 1)
    ax.set_xscale("log")

    ax.plot(
        10 ** logsm_bins[1:], mean_fsat01, color=MBLUE, label=rf"$z={z_bin[0]:.1f}$"
    )
    ax.plot(
        10 ** logsm_bins[1:], mean_fsat12, color=MGREEN, label=rf"$z={z_bin[1]:.1f}$"
    )
    ax.plot(10 ** logsm_bins[1:], mean_fsat23, color=MRED, label=rf"$z={z_bin[2]:.1f}$")

    ax.legend()
    xlabel = ax.set_xlabel(r"$M_{\star}\ [M_{\odot}]$")
    ylabel = ax.set_ylabel(r"${\rm satellite\ fraction}$")
    fn_out = os.path.join(drn_out, "fsat_analysis_cosmos_260316_04_26_2026.png")
    fig.savefig(
        fn_out, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )

    plt.close()
    return fig


def plot_hod(*, pdata, logsm_samples=[10, 11], z_plot=0.5, dz=0.5, drn_out=""):
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG
    assert HAS_ASTROPY, ASTROPY_MSG

    drn_out = drn_out or "."
    os.makedirs(drn_out, exist_ok=True)

    logmp_bins = np.linspace(11, 14.75, 50)

    msk_cen = pdata["central"] == 1
    msk_sat = ~msk_cen

    msk_zplot = np.abs(pdata["redshift_true"] - z_plot) < dz

    halo_counts, __ = np.histogram(
        pdata["logmp_obs_host"][msk_cen & msk_zplot], bins=logmp_bins
    )

    fig, ax = plt.subplots(1, 1)
    ax.loglog()

    logsm_cut = logsm_samples[0]
    msk_logsm = pdata["logsm_obs"] > logsm_cut
    cen_counts_10, __ = np.histogram(
        pdata["logmp_obs_host"][msk_cen & msk_logsm & msk_zplot], bins=logmp_bins
    )
    sat_counts_10, __ = np.histogram(
        pdata["logmp_obs_host"][msk_sat & msk_logsm & msk_zplot], bins=logmp_bins
    )
    ax.plot(
        10 ** logmp_bins[1:],
        cen_counts_10 / halo_counts,
        label=r"${\rm centrals}$",
        color=MBLUE,
    )
    ax.plot(
        10 ** logmp_bins[1:],
        sat_counts_10 / halo_counts,
        label=r"${\rm satellites}$",
        color=MRED,
    )

    logsm_cut = logsm_samples[1]
    msk_logsm = pdata["logsm_obs"] > logsm_cut
    cen_counts_11, __ = np.histogram(
        pdata["logmp_obs_host"][msk_cen & msk_logsm & msk_zplot], bins=logmp_bins
    )
    sat_counts_11, __ = np.histogram(
        pdata["logmp_obs_host"][msk_sat & msk_logsm & msk_zplot], bins=logmp_bins
    )
    ax.plot(10 ** logmp_bins[1:], cen_counts_11 / halo_counts, "--", color=MBLUE)
    ax.plot(10 ** logmp_bins[1:], sat_counts_11 / halo_counts, "--", color=MRED)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(5e-4, ymax)

    xlabel = ax.set_xlabel(r"$M_{\rm halo}\ [M_{\odot}]$")
    ylabel = ax.set_ylabel(r"$\langle N_{\rm gal}\rangle$")

    red_line = mlines.Line2D([], [], ls="-", c=MRED, label=r"${\rm satellites}$")
    blue_line = mlines.Line2D([], [], ls="-", c=MBLUE, label=r"${\rm centrals}$")
    leg0 = ax.legend(handles=[red_line, blue_line], loc="upper left")

    solid_line = mlines.Line2D(
        [], [], ls="-", c="gray", label=r"$M_{\star}>10^{10}M_{\odot}$"
    )
    dashed_line = mlines.Line2D(
        [], [], ls="--", c="gray", label=r"$M_{\star}>10^{11}M_{\odot}$"
    )
    ax.add_artist(leg0)
    ax.legend(handles=[solid_line, dashed_line], loc="lower right")

    fn_out = os.path.join(drn_out, "hod_analysis_cosmos_260316_04_26_2026.png")

    fig.savefig(
        fn_out, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )
    plt.close()
    return fig


def plot_csmf_cens(*, pdata, drn_out=""):
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG
    assert HAS_ASTROPY, ASTROPY_MSG

    drn_out = drn_out or "."
    os.makedirs(drn_out, exist_ok=True)

    fig, ax = plt.subplots(1, 1)
    ax.set_xscale("log")
    logsm_bins = np.linspace(9, 13, 30)

    msk_cen = pdata["central"] == 1

    lo, hi = 12, 12.25
    msk_lgmhost = (pdata["logmp_obs_host"] > lo) & (pdata["logmp_obs_host"] < hi)
    y, x = np.histogram(
        pdata["logsm_obs"][msk_lgmhost & msk_cen], bins=logsm_bins, density=True
    )
    ax.fill_between(
        10 ** x[1:],
        np.zeros_like(y),
        y,
        color=MBLUE,
        alpha=0.5,
        label=r"$M_{\rm halo}\approx10^{12}M_{\odot}$",
    )

    lo, hi = 13, 13.25
    msk_lgmhost = (pdata["logmp_obs_host"] > lo) & (pdata["logmp_obs_host"] < hi)
    y, x = np.histogram(
        pdata["logsm_obs"][msk_lgmhost & msk_cen], bins=logsm_bins, density=True
    )
    ax.fill_between(
        10 ** x[1:],
        np.zeros_like(y),
        y,
        color=MORANGE,
        alpha=0.5,
        label=r"$M_{\rm halo}\approx10^{13}M_{\odot}$",
    )

    lo, hi = 14, 14.25
    msk_lgmhost = (pdata["logmp_obs_host"] > lo) & (pdata["logmp_obs_host"] < hi)
    y, x = np.histogram(
        pdata["logsm_obs"][msk_lgmhost & msk_cen], bins=logsm_bins, density=True
    )
    ax.fill_between(
        10 ** x[1:],
        np.zeros_like(y),
        y,
        color=MRED,
        alpha=0.5,
        label=r"$M_{\rm halo}\approx10^{14}M_{\odot}$",
    )

    ax.legend(loc="upper right")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0.01, 2)

    xlabel = ax.set_xlabel(r"$M_{\star}\ [M_{\odot}]$")
    ylabel = ax.set_ylabel(r"$P(M_{\star}\vert M_{\rm halo})$")
    fn_out = os.path.join(drn_out, "cen_csmf_analysis_cosmos_260316_04_26_2026.png")

    fig.savefig(
        fn_out, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )
    plt.close()
    return fig


def plot_csmf_sats(*, pdata, logsm_cut=9.0, drn_out=""):
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG
    assert HAS_ASTROPY, ASTROPY_MSG

    drn_out = drn_out or "."
    os.makedirs(drn_out, exist_ok=True)

    fig, ax = plt.subplots(1, 1)
    logsm_bins = np.linspace(logsm_cut, 12, 50)
    __ = ax.loglog()

    msk_logsm = pdata["logsm_obs"] > logsm_cut
    msk_cen = pdata["central"] == 1
    msk_sat = ~msk_cen

    lo, hi = 14, 15.5
    msk_lgmhost = (pdata["logmp_obs_host"] > lo) & (pdata["logmp_obs_host"] < hi)
    y, __ = np.histogram(
        pdata["logsm_obs"][msk_lgmhost & msk_sat & msk_logsm], bins=logsm_bins
    )
    __ = ax.plot(
        10 ** logsm_bins[1:],
        y / (np.sum(msk_lgmhost & msk_cen)),
        drawstyle="steps",
        color=MRED,
        label=r"$M_{\rm host}\approx10^{14.5}M_{\odot}$",
    )

    lo, hi = 13, 13.25
    msk_lgmhost = (pdata["logmp_obs_host"] > lo) & (pdata["logmp_obs_host"] < hi)
    y, __ = np.histogram(
        pdata["logsm_obs"][msk_lgmhost & msk_sat & msk_logsm], bins=logsm_bins
    )
    __ = ax.plot(
        10 ** logsm_bins[1:],
        y / (np.sum(msk_lgmhost & msk_cen)),
        drawstyle="steps",
        color=MORANGE,
        label=r"$M_{\rm host}\approx10^{13}M_{\odot}$",
    )

    lo, hi = 12, 12.25
    msk_lgmhost = (pdata["logmp_obs_host"] > lo) & (pdata["logmp_obs_host"] < hi)
    y, __ = np.histogram(
        pdata["logsm_obs"][msk_lgmhost & msk_sat & msk_logsm], bins=logsm_bins
    )
    ax.plot(
        10 ** logsm_bins[1:],
        y / (np.sum(msk_lgmhost & msk_cen)),
        drawstyle="steps",
        color=MBLUE,
        label=r"$M_{\rm host}\approx10^{12}M_{\odot}$",
    )

    ax.legend()
    xlabel = ax.set_xlabel(r"$M_{\star}\ [M_{\odot}]$")
    ylabel = ax.set_ylabel(r"$\Phi(M_{\star}\vert M_{\rm host})$")

    fn_out = os.path.join(drn_out, "sat_csmf_analysis_cosmos_260316_04_26_2026.png")

    fig.savefig(
        fn_out, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )
    plt.close()
    return fig
