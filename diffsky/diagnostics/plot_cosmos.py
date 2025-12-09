""""""

import os
from collections import namedtuple
from functools import lru_cache

import numpy as np
from dsps.data_loaders import load_ssp_templates, load_transmission_curve
from jax import random as jran

from ..data_loaders import cosmos20_loader as c20
from ..experimental.mc_lightcone_halos import mc_weighted_lightcone_data
from ..experimental.mc_phot import mc_lc_phot

MBLUE = "#1f77b4"
MGREEN = "#2ca02c"
MRED = "#d62728"

try:
    from matplotlib import lines as mlines
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

COSMOS_FILTER_BNAMES = (
    "g_HSC",
    "r_HSC",
    "i_HSC",
    "z_HSC",
    "y_HSC",
    "Y_uv",
    "J_uv",
    "H_uv",
    "K_uv",
)

MAG_I_LO, MAG_I_HI = 18.5, 25.5
MAG_BINS = mag_bins = np.linspace(MAG_I_LO, MAG_I_HI, 30)


@lru_cache()
def get_plotting_data(seed):
    ran_key = jran.key(seed)

    cosmos = c20.load_cosmos20()
    cosmos = c20.apply_nan_cuts(cosmos)

    msk_is_complete = c20.get_is_complete_mask(cosmos)
    cosmos = cosmos[msk_is_complete]

    msk_is_not_hsc_outlier = c20.get_color_outlier_mask(cosmos, c20.HSC_MAG_NAMES)
    msk_is_not_uvista_outlier = c20.get_color_outlier_mask(cosmos, c20.UVISTA_MAG_NAMES)
    msk_is_not_uvista_outlier.mean(), msk_is_not_hsc_outlier.mean()
    cosmos = cosmos[msk_is_not_hsc_outlier & msk_is_not_uvista_outlier]

    tcurves = _get_cosmos_dsps_tcurves()

    ssp_data = load_ssp_templates()

    filter_dict = dict()
    for i, cosmos_key in enumerate(c20.COSMOS_TARGET_MAGS):
        filter_dict[cosmos_key] = i

    num_halos = 20_000
    z_min, z_max = cosmos["photoz"].min(), cosmos["photoz"].max()
    lgmp_min, lgmp_max = 10.0, 15.0
    sky_area_degsq = 100.0 * c20.SKY_AREA

    n_z_phot_table = 15
    z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

    halo_lc_data = (num_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)
    phot_data = (ssp_data, tcurves, z_phot_table)

    ran_key, lc_halo_key = jran.split(ran_key, 2)
    args = (lc_halo_key, *halo_lc_data, *phot_data)

    lc_data = mc_weighted_lightcone_data(*args)

    ran_key, sed_key = jran.split(ran_key, 2)
    diffsky_data = mc_lc_phot(sed_key, lc_data)
    diffsky_data["filter_dict"] = filter_dict
    diffsky_data["sky_area_degsq"] = sky_area_degsq

    PlottingData = namedtuple("PlottingData", ("cosmos", "lc_data", "diffsky_data"))
    pdata = PlottingData(cosmos, lc_data, diffsky_data)

    return pdata


def _get_cosmos_dsps_tcurves(bnames=COSMOS_FILTER_BNAMES):
    tcurves = []
    for bn_pat in bnames:
        tcurve = load_transmission_curve(bn_pat=bn_pat + "*")
        tcurves.append(tcurve)
    return tcurves


def plot_app_mag_func(
    z_bin,
    dz=0.2,
    mag_bins=MAG_BINS,
    m0=c20.HSC_MAG_NAMES[0],
    m1=c20.HSC_MAG_NAMES[2],
    m2=c20.UVISTA_MAG_NAMES[1],
    drn_out="",
):
    if not HAS_MATPLOTLIB:
        raise ImportError("Must have matplotlib installed to make diagnostic plots")

    os.makedirs(drn_out, exist_ok=True)

    pdata = get_plotting_data(0)

    fig, ax = plt.subplots(1, 1)
    ax.set_yscale("log")

    # ax.set_xlim(26, 18.0)
    # ax.set_ylim(1, 1e4)

    xlabel = ax.set_xlabel(r"$m_{\rm x}$")
    ylabel = ax.set_ylabel(r"$\phi(m)$")

    mag_binmids = 0.5 * (mag_bins[:-1] + mag_bins[1:])

    ax.set_title(f"z={z_bin:.1f}")

    msk_z = np.abs(pdata.cosmos["photoz"] - z_bin) < dz
    msk_z_pred = np.abs(pdata.lc_data.z_obs - z_bin) < dz

    target = np.histogram(pdata.cosmos[m0][msk_z], bins=mag_bins)[0] / c20.SKY_AREA
    pred = (
        np.histogram(
            pdata.diffsky_data["obs_mags"][:, pdata.diffsky_data["filter_dict"][m0]][
                msk_z_pred
            ],
            bins=mag_bins,
            weights=pdata.lc_data.nhalos[msk_z_pred],
        )[0]
        / pdata.diffsky_data["sky_area_degsq"]
    )
    __ = ax.plot(mag_binmids, target, color=MBLUE, label=r"${\rm g}$")
    __ = ax.plot(mag_binmids, pred, "--", color=MBLUE)

    target = np.histogram(pdata.cosmos[m1][msk_z], bins=mag_bins)[0] / c20.SKY_AREA
    pred = (
        np.histogram(
            pdata.diffsky_data["obs_mags"][:, pdata.diffsky_data["filter_dict"][m1]][
                msk_z_pred
            ],
            bins=mag_bins,
            weights=pdata.lc_data.nhalos[msk_z_pred],
        )[0]
        / pdata.diffsky_data["sky_area_degsq"]
    )
    __ = ax.plot(mag_binmids, target, color=MGREEN, label=r"${\rm i}$")
    __ = ax.plot(mag_binmids, pred, "--", color=MGREEN)

    target = np.histogram(pdata.cosmos[m2][msk_z], bins=mag_bins)[0] / c20.SKY_AREA
    pred = (
        np.histogram(
            pdata.diffsky_data["obs_mags"][:, pdata.diffsky_data["filter_dict"][m2]][
                msk_z_pred
            ],
            bins=mag_bins,
            weights=pdata.lc_data.nhalos[msk_z_pred],
        )[0]
        / pdata.diffsky_data["sky_area_degsq"]
    )
    __ = ax.plot(mag_binmids, target, color=MRED, label=r"${\rm J}$")
    __ = ax.plot(mag_binmids, pred, "--", color=MRED)

    blue_line = mlines.Line2D([], [], ls="-", c=MBLUE, label=m0)
    green_line = mlines.Line2D([], [], ls="-", c=MGREEN, label=m1)
    red_line = mlines.Line2D([], [], ls="-", c=MRED, label=m2)

    solid_line = mlines.Line2D([], [], ls="-", c="k", label=r"${\rm COSMOS}$")
    dashed_line = mlines.Line2D([], [], ls="--", c="k", label=r"${\rm Diffsky}$")

    leg0 = ax.legend(handles=[red_line, green_line, blue_line], loc="upper right")
    ax.add_artist(leg0)
    ax.legend(handles=[dashed_line, solid_line], loc="lower left")

    bn_out = f"cosmos_vs_diffsky_app_mag_func_z={z_bin:.1f}.png"
    fn_out = os.path.join(drn_out, bn_out)
    fig.savefig(
        fn_out, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )
    return fig


def make_another_plot():
    pdata = get_plotting_data(0)
    # make plot


def make_third_plot():
    pdata = get_plotting_data(1)
    # make plot
