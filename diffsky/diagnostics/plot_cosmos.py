"""Module to make diagnostic plots comparing COSMOS-20 to diffsky lightcones"""

import os
import random
from collections import namedtuple
from functools import lru_cache
from glob import glob

import numpy as np
from diffmah import DEFAULT_MAH_PARAMS, logmh_at_t_obs
from diffstar.defaults import FB
from dsps import data_loaders as ddl
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import DEFAULT_COSMOLOGY, flat_wcdm
from dsps.data_loaders.load_filter_data import TransmissionCurve
from dsps.data_loaders.retrieve_fake_fsps_data import load_fake_ssp_data
from jax import random as jran
from jax.scipy.stats import norm

from ..data_loaders import cosmos20_loader as c20
from ..data_loaders.hacc_utils import lightcone_utils as hlu
from ..data_loaders.hacc_utils import load_lc_mock
from ..experimental import precompute_ssp_phot as psspp
from ..experimental.mc_lightcone_halos import mc_weighted_lightcone_data
from ..experimental.mc_phot import mc_lc_phot
from ..param_utils import diffsky_param_wrapper as dpw
from ..phot_utils import get_wave_eff_table

MBLUE = "#1f77b4"
MGREEN = "#2ca02c"
MRED = "#d62728"

N_SFH_TABLE = 100

try:
    from matplotlib import lines as mlines
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
MATPLOTLIB_MSG = "Must have matplotlib installed to use this function"

try:
    from astropy.table import Table, vstack

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

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
COLOR_PAIRS = (
    ("HSC_g_MAG", "HSC_r_MAG"),
    ("HSC_r_MAG", "HSC_i_MAG"),
    ("HSC_i_MAG", "HSC_z_MAG"),
    ("UVISTA_J_MAG", "UVISTA_H_MAG"),
    ("UVISTA_H_MAG", "UVISTA_Ks_MAG"),
)

MAG_I_LO, MAG_I_HI = 18.5, 25.5
MAG_BINS = mag_bins = np.linspace(MAG_I_LO, MAG_I_HI, 30)

Z_MAGI_PAIRS = (
    (0.6, 20.0),
    (0.6, 22.0),
    (0.6, 24.0),
    (1.0, 22.0),
    (1.0, 24.0),
    (2.0, 22.0),
    (2.0, 24.0),
)


@lru_cache()
def get_plotting_data_mock(
    *,
    drn_mock,
    drn_synthetic_cores,
    lc_patch,
    seed=0,
    diffstarpop_params=None,
    mzr_params=None,
    spspop_params=None,
    scatter_params=None,
    ssperr_params=None,
    fb=None,
    cosmos=None,
    tcurves=None,
    ssp_data=None,
    num_halos=200_000,
    sky_area_degsq=None,
    z_min=c20.Z_MIN,
    z_max=c20.Z_MAX,
    n_sfh_table=N_SFH_TABLE,
):
    """Generate lightcone halos and galaxy photometry
    to make diagnostic plots for the input model

    Returns
    -------
    cosmos : astropy.table.Table
        COSMOS-20 dataset

    lc_data : namedtuple
        Halo lightcone data

    diffsky_data : dict
        Dictionary of diffsky galaxies with photometry and associated data

    """
    ran_key = jran.key(seed)

    keys = (
        "logsm_obs",
        "logmp_obs",
        "redshift_true",
        "lsst_i",
        "vx",
        "vy",
        "vz",
        "central",
        *DEFAULT_MAH_PARAMS._fields,
    )

    bnpat = f"lc_cores-*.{lc_patch}.diffsky_gals.hdf5"
    fn_list = sorted(glob(os.path.join(drn_mock, bnpat)))
    fn_list_synthetic = glob(os.path.join(drn_synthetic_cores, bnpat))

    metadata = load_lc_mock.load_mock_metadata(fn_list[0])
    if ssp_data is None:
        ssp_data = metadata["ssp_data"]
    elif ssp_data == "random":
        ssp_data = load_fake_ssp_data()

    dsps_cosmo_mock = DEFAULT_COSMOLOGY._make(
        [metadata["cosmology"][key] for key in DEFAULT_COSMOLOGY._fields]
    )
    if fb is None:
        fb = metadata["cosmology"]["Ob0"] / metadata["cosmology"]["Om0"]

    if diffstarpop_params is None:
        diffstarpop_params = dpw.DEFAULT_PARAM_COLLECTION.diffstarpop_params._make(
            metadata["param_collection"].diffstarpop_params
        )
    if mzr_params is None:
        mzr_params = dpw.DEFAULT_PARAM_COLLECTION.mzr_params._make(
            metadata["param_collection"].mzr_params
        )
    if spspop_params is None:
        spspop_params = dpw.DEFAULT_PARAM_COLLECTION.spspop_params._make(
            metadata["param_collection"].spspop_params
        )
    if scatter_params is None:
        scatter_params = dpw.DEFAULT_PARAM_COLLECTION.scatter_params._make(
            metadata["param_collection"].scatter_params
        )
    if ssperr_params is None:
        ssperr_params = dpw.DEFAULT_PARAM_COLLECTION.ssperr_params._make(
            metadata["param_collection"].ssperr_params
        )

    mock = Table(load_lc_mock.load_lc_patch_collection(fn_list, keys))
    mock["synthetic"] = False
    mock2 = Table(load_lc_mock.load_lc_patch_collection(fn_list_synthetic, keys))
    mock2["synthetic"] = True

    n_tot = len(mock) + len(mock2)
    downsample_factor = n_tot // num_halos
    mock = vstack((mock, mock2))[::downsample_factor]
    n_mock = len(mock)

    mock["t_obs"] = flat_wcdm.age_at_z(
        np.array(mock["redshift_true"]), *dsps_cosmo_mock
    )
    mah_params = DEFAULT_MAH_PARAMS._make(
        [np.array(mock[key]) for key in DEFAULT_MAH_PARAMS._fields]
    )
    t0 = flat_wcdm.age_at_z0(*dsps_cosmo_mock)
    lgt0 = np.log10(t0)
    logmp0 = logmh_at_t_obs(mah_params, np.zeros(n_mock) + t0, lgt0)
    logmp_obs = logmh_at_t_obs(
        mah_params, np.zeros(n_mock) + np.array(mock["t_obs"]), lgt0
    )

    t_table = np.linspace(T_TABLE_MIN, t0, n_sfh_table)

    n_z_phot_table = 25
    EPS = 1e-3
    z_min = max(mock["redshift_true"].min() - EPS, EPS)
    z_max = mock["redshift_true"].max() + EPS
    z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

    if tcurves is None:
        tcurves = _get_cosmos_dsps_tcurves()
    elif tcurves == "random":
        tcurves = _get_random_tcurves()

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, dsps_cosmo_mock
    )
    wave_eff_table = get_wave_eff_table(z_phot_table, tcurves)

    lc_data_dict = dict(
        nhalos=np.ones(n_mock).astype(float) * downsample_factor,
        z_obs=np.array(mock["redshift_true"]),
        t_obs=np.array(mock["t_obs"]),
        mah_params=mah_params,
        logmp0=logmp0,
        t_table=t_table,
        ssp_data=ssp_data,
        precomputed_ssp_mag_table=precomputed_ssp_mag_table,
        z_phot_table=z_phot_table,
        wave_eff_table=wave_eff_table,
        logmp_obs=logmp_obs,
    )
    LCData = namedtuple("LCData", list(lc_data_dict.keys()))
    lc_data = LCData(**lc_data_dict)

    if cosmos is None:
        cosmos = c20.load_cosmos20()
    elif cosmos == "random":
        ran_key, cosmos_key = jran.split(ran_key, 2)
        cosmos = _generate_random_cosmos_data(cosmos_key)

    cosmos = c20.apply_nan_cuts(cosmos)

    msk_is_complete = c20.get_is_complete_mask(cosmos)
    cosmos = cosmos[msk_is_complete]

    msk_is_not_hsc_outlier = c20.get_color_outlier_mask(cosmos, c20.HSC_MAG_NAMES)
    msk_is_not_uvista_outlier = c20.get_color_outlier_mask(cosmos, c20.UVISTA_MAG_NAMES)
    msk_is_not_uvista_outlier.mean(), msk_is_not_hsc_outlier.mean()
    cosmos = cosmos[msk_is_not_hsc_outlier & msk_is_not_uvista_outlier]

    filter_dict = dict()
    for i, cosmos_key in enumerate(c20.COSMOS_TARGET_MAGS):
        filter_dict[cosmos_key] = i, COSMOS_FILTER_BNAMES[i]

    ran_key, sed_key = jran.split(ran_key, 2)
    diffsky_data = mc_lc_phot(
        sed_key,
        lc_data,
        diffstarpop_params=diffstarpop_params,
        mzr_params=mzr_params,
        spspop_params=spspop_params,
        scatter_params=scatter_params,
        ssperr_params=ssperr_params,
        cosmo_params=dsps_cosmo_mock,
        fb=fb,
    )
    diffsky_data["filter_dict"] = filter_dict
    _res = hlu.read_hacc_lc_patch_decomposition(metadata["nbody_info"]["sim_name"])
    patch_decomposition, sky_frac, solid_angles = _res
    diffsky_data["sky_area_degsq"] = solid_angles[lc_patch]

    PlottingData = namedtuple("PlottingData", ("cosmos", "lc_data", "diffsky_data"))
    pdata = PlottingData(cosmos, lc_data, diffsky_data)

    return pdata


@lru_cache()
def get_plotting_data(
    *,
    seed=0,
    diffstarpop_params=dpw.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
    mzr_params=dpw.DEFAULT_PARAM_COLLECTION.mzr_params,
    spspop_params=dpw.DEFAULT_PARAM_COLLECTION.spspop_params,
    scatter_params=dpw.DEFAULT_PARAM_COLLECTION.scatter_params,
    ssperr_params=dpw.DEFAULT_PARAM_COLLECTION.ssperr_params,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    cosmos=None,
    tcurves=None,
    ssp_data=None,
    num_halos=20_000,
    lc_data=None,
    sky_area_degsq=None,
    z_min=c20.Z_MIN,
    z_max=c20.Z_MAX,
    skip_param_check=False,
):
    """Generate lightcone halos and galaxy photometry
    to make diagnostic plots for the input model

    Returns
    -------
    cosmos : astropy.table.Table
        COSMOS-20 dataset

    lc_data : namedtuple
        Halo lightcone data

    diffsky_data : dict
        Dictionary of diffsky galaxies with photometry and associated data

    """
    ran_key = jran.key(seed)

    if cosmos is None:
        cosmos = c20.load_cosmos20()
    elif cosmos == "random":
        ran_key, cosmos_key = jran.split(ran_key, 2)
        cosmos = _generate_random_cosmos_data(cosmos_key)

    cosmos = c20.apply_nan_cuts(cosmos)

    msk_is_complete = c20.get_is_complete_mask(cosmos)
    cosmos = cosmos[msk_is_complete]

    msk_is_not_hsc_outlier = c20.get_color_outlier_mask(cosmos, c20.HSC_MAG_NAMES)
    msk_is_not_uvista_outlier = c20.get_color_outlier_mask(cosmos, c20.UVISTA_MAG_NAMES)
    msk_is_not_uvista_outlier.mean(), msk_is_not_hsc_outlier.mean()
    cosmos = cosmos[msk_is_not_hsc_outlier & msk_is_not_uvista_outlier]

    if tcurves is None:
        tcurves = _get_cosmos_dsps_tcurves()
    elif tcurves == "random":
        tcurves = _get_random_tcurves()

    if ssp_data is None:
        ssp_data = ddl.load_ssp_templates()
    elif ssp_data == "random":
        ssp_data = load_fake_ssp_data()

    filter_dict = dict()
    for i, cosmos_key in enumerate(c20.COSMOS_TARGET_MAGS):
        filter_dict[cosmos_key] = i, COSMOS_FILTER_BNAMES[i]

    if lc_data is None:
        lgmp_min, lgmp_max = 10.0, 15.0
        sky_area_degsq = 100.0 * c20.SKY_AREA

        n_z_phot_table = 15
        z_min, z_max = cosmos["photoz"].min(), cosmos["photoz"].max()
        z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

        halo_lc_data = (num_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)
        phot_data = (ssp_data, tcurves, z_phot_table)

        ran_key, lc_halo_key = jran.split(ran_key, 2)
        args = (lc_halo_key, *halo_lc_data, *phot_data)
        lc_data = mc_weighted_lightcone_data(*args)
    else:
        assert z_min is not None
        assert z_max is not None
        assert sky_area_degsq is not None

    ran_key, sed_key = jran.split(ran_key, 2)
    diffsky_data = mc_lc_phot(
        sed_key,
        lc_data,
        diffstarpop_params=diffstarpop_params,
        mzr_params=mzr_params,
        spspop_params=spspop_params,
        scatter_params=scatter_params,
        ssperr_params=ssperr_params,
        cosmo_params=cosmo_params,
        fb=fb,
        skip_param_check=skip_param_check,
    )
    diffsky_data["filter_dict"] = filter_dict
    diffsky_data["sky_area_degsq"] = sky_area_degsq

    PlottingData = namedtuple("PlottingData", ("cosmos", "lc_data", "diffsky_data"))
    pdata = PlottingData(cosmos, lc_data, diffsky_data)

    return pdata


def _generate_random_cosmos_data(ran_key, n_gals=2_000):
    if not HAS_ASTROPY:
        msg = "Must have astropy installed to use _generate_random_cosmos_data"
        raise ImportError(msg)

    cosmos = Table()
    ran_key, redshift_key = jran.split(ran_key, 2)
    cosmos["photoz"] = jran.uniform(
        redshift_key, minval=c20.Z_MIN, maxval=c20.Z_MAX, shape=(n_gals,)
    )
    for key in c20.COSMOS_TARGET_MAGS:
        ran_key, col_key = jran.split(ran_key, 2)
        ran_data = jran.uniform(col_key, minval=15, maxval=28, shape=(n_gals,))
        cosmos[key] = ran_data
    return cosmos


def _get_cosmos_dsps_tcurves(bnames=COSMOS_FILTER_BNAMES):
    tcurves = []
    for bn_pat in bnames:
        tcurve = ddl.load_transmission_curve(bn_pat=bn_pat + "*")
        tcurves.append(tcurve)

    Tcurves = namedtuple("TCurves", COSMOS_FILTER_BNAMES)
    return Tcurves(*tcurves)


def _get_random_tcurves(bnames=COSMOS_FILTER_BNAMES):
    tcurves = []
    for __ in bnames:
        tcurve = _load_random_transmission_curve()
        tcurves.append(tcurve)

    Tcurves = namedtuple("TCurves", COSMOS_FILTER_BNAMES)
    return Tcurves(*tcurves)


def _load_random_transmission_curve(
    ran_key=None, tcurve_center=None, wave_range=(1_000, 10_000), scale=300
):
    """Reimplementation from dsps.data_loaders.load_filter_data"""
    if ran_key is None:
        seed = random.randint(0, 2**32 - 1)
        ran_key = jran.key(seed)

    if tcurve_center is None:
        xmin = wave_range[0] + scale * 2
        xmax = wave_range[1] - scale * 2
        tcurve_center = jran.uniform(ran_key, minval=xmin, maxval=xmax)
    else:
        assert wave_range[0] < tcurve_center < wave_range[1]

    wave = np.linspace(*wave_range, 200)

    _transmission = norm.pdf(wave, loc=tcurve_center, scale=scale)
    transmission = _transmission / _transmission.max()
    tcurve = TransmissionCurve(wave, transmission)

    return tcurve


def plot_app_mag_func(
    *,
    pdata,
    z_bin,
    dz=0.25,
    mag_bins=MAG_BINS,
    m0=c20.HSC_MAG_NAMES[0],
    m1=c20.HSC_MAG_NAMES[2],
    m2=c20.UVISTA_MAG_NAMES[1],
    drn_out="",
    model_nickname="default",
):
    """Plot a comparison to the COSMOS apparent magnitude function
    of the input diffsky galaxy population.

    Parameters
    ----------
    pdata : namedtuple
        Data created by the plot_cosmos.get_plotting_data function

    z_bin : float
        Redshift of the galaxy population to plot

    m0, m1, m2 : strings
        Columns of the COSMOS dataset used to check the apparent magnitude function
        Defaults:
            m0 = 'HSC_g_MAG'
            m1 = 'HSC_i_MAG'
            m2 = 'UVISTA_J_MAG'

    drn_out : string
        Output directory to store plots

    model_nickname : string
        Nickname of the model being tested. Default is `default`.
        The model_nickname will be part of the output filename of the plot.

    """
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG

    os.makedirs(drn_out, exist_ok=True)

    m0_label = pdata.diffsky_data["filter_dict"][m0][1].split("_")[0]
    m1_label = pdata.diffsky_data["filter_dict"][m1][1].split("_")[0]
    m2_label = pdata.diffsky_data["filter_dict"][m2][1].split("_")[0]

    fig, ax = plt.subplots(1, 1)
    ax.set_yscale("log")

    ax.set_xlim(mag_bins.max() + 0.5, mag_bins.min() - 0.5)

    xlabel = ax.set_xlabel(r"${\rm mag}$")
    ylabel = ax.set_ylabel(r"$\phi(m)\ {\rm [deg^{-2}}]$")

    mag_binmids = 0.5 * (mag_bins[:-1] + mag_bins[1:])
    dmagbins = np.diff(mag_bins)

    ax.set_title(f"z={z_bin:.1f}")

    msk_z = np.abs(pdata.cosmos["photoz"] - z_bin) < dz
    msk_z_pred = np.abs(pdata.lc_data.z_obs - z_bin) < dz

    w_pred = pdata.lc_data.nhalos[msk_z_pred]
    pred_norm_factor = pdata.diffsky_data["sky_area_degsq"] * dmagbins * 2 * dz
    target_norm_factor = c20.SKY_AREA * dmagbins * 2 * dz

    _target = np.histogram(pdata.cosmos[m0][msk_z], bins=mag_bins)[0]
    target_nd_m0 = _target / target_norm_factor
    indx_m0 = pdata.diffsky_data["filter_dict"][m0][0]
    x_m0 = pdata.diffsky_data["obs_mags"][:, indx_m0][msk_z_pred]
    pred_counts_m0 = np.histogram(x_m0, bins=mag_bins, weights=w_pred)[0]
    pred_nd_m0 = pred_counts_m0 / pred_norm_factor
    ax.plot(mag_binmids, target_nd_m0, color=MBLUE, label=m0_label)
    ax.plot(mag_binmids, pred_nd_m0, "--", color=MBLUE)

    _target = np.histogram(pdata.cosmos[m1][msk_z], bins=mag_bins)[0]
    target_nd_m1 = _target / target_norm_factor
    indx_m1 = pdata.diffsky_data["filter_dict"][m1][0]
    x_m1 = pdata.diffsky_data["obs_mags"][:, indx_m1][msk_z_pred]
    pred_counts_m1 = np.histogram(x_m1, bins=mag_bins, weights=w_pred)[0]
    pred_nd_m1 = pred_counts_m1 / pred_norm_factor

    ax.plot(mag_binmids, target_nd_m1, color=MGREEN, label=m1_label)
    ax.plot(mag_binmids, pred_nd_m1, "--", color=MGREEN)

    _target = np.histogram(pdata.cosmos[m2][msk_z], bins=mag_bins)[0]
    target_nd_m2 = _target / target_norm_factor
    indx_m2 = pdata.diffsky_data["filter_dict"][m2][0]
    x_m2 = pdata.diffsky_data["obs_mags"][:, indx_m2][msk_z_pred]
    pred_counts_m2 = np.histogram(x_m2, bins=mag_bins, weights=w_pred)[0]
    pred_nd_m2 = pred_counts_m2 / pred_norm_factor

    ax.plot(mag_binmids, target_nd_m2, color=MRED, label=m2_label)
    ax.plot(mag_binmids, pred_nd_m2, "--", color=MRED)

    blue_line = mlines.Line2D([], [], ls="-", c=MBLUE, label=m0_label)
    green_line = mlines.Line2D([], [], ls="-", c=MGREEN, label=m1_label)
    red_line = mlines.Line2D([], [], ls="-", c=MRED, label=m2_label)

    solid_line = mlines.Line2D([], [], ls="-", c="k", label=r"${\rm COSMOS}$")
    dashed_line = mlines.Line2D([], [], ls="--", c="k", label=r"${\rm Diffsky}$")

    leg0 = ax.legend(handles=[red_line, green_line, blue_line], loc="upper right")
    ax.add_artist(leg0)
    ax.legend(handles=[dashed_line, solid_line], loc="lower left")

    bn_out = f"{model_nickname}_diffsky_vs_cosmos_app_mag_func_z={z_bin:.1f}.png"
    fn_out = os.path.join(drn_out, bn_out)
    fig.savefig(
        fn_out, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )
    plt.close()
    return fig


def plot_color_pdf(
    *,
    pdata,
    z_bin,
    m1_bin,
    c0,
    c1,
    dz=0.25,
    m1=c20.HSC_MAG_NAMES[2],
    drn_out="",
    model_nickname="default",
):
    """Plot a comparison to the COSMOS color PDF of the input diffsky galaxy population

    Parameters
    ----------
    pdata : namedtuple
        Data created by the plot_cosmos.get_plotting_data function

    z_bin : float
        Redshift of the galaxy population to plot

    m1_bin : float
        Apparent of the galaxy population to plot
        Default magnitude column is m1=`HSC_i_MAG`

    m1 : string
        Column name of the magnitude used to bin the galaxies
        Default is m1=`HSC_i_MAG`

    c0, c1 : strings
        Columns of the COSMOS dataset used to define the plotted color
        Examples: 'HSC_g_MAG', 'HSC_i_MAG', 'UVISTA_H_MAG',  'UVISTA_Ks_MAG'

    drn_out : string
        Output directory to store plots

    model_nickname : string
        Nickname of the model being tested. Default is `default`.
        The model_nickname will be part of the output filename of the plot.

    """
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG

    os.makedirs(drn_out, exist_ok=True)

    fig, ax = plt.subplots(1, 1)

    m1_label = pdata.diffsky_data["filter_dict"][m1][1].split("_")[0]
    c0_label = pdata.diffsky_data["filter_dict"][c0][1].split("_")[0]
    c1_label = pdata.diffsky_data["filter_dict"][c1][1].split("_")[0]

    xlabel = ax.set_xlabel(f"{c0_label}-{c1_label}")
    ylabel = ax.set_ylabel(r"{\rm PDF}")

    msk_z = np.abs(pdata.cosmos["photoz"] - z_bin) < dz
    msk_z_pred = np.abs(pdata.lc_data.z_obs - z_bin) < dz

    msk_mag = np.abs(pdata.cosmos[m1] - m1_bin) < 1
    indx_m1 = pdata.diffsky_data["filter_dict"][m1][0]
    msk_mag_pred = np.abs(pdata.diffsky_data["obs_mags"][:, indx_m1] - m1_bin) < 1

    msk_sample = msk_z & msk_mag
    msk_sample_pred = msk_z_pred & msk_mag_pred

    color_target = (pdata.cosmos[c0] - pdata.cosmos[c1])[msk_sample]
    __, target_bins, __ = ax.hist(
        color_target, bins=50, label=r"${\rm COSMOS}$", density=True, alpha=0.7
    )

    w_pred = pdata.lc_data.nhalos[msk_sample_pred]

    indx_c0 = pdata.diffsky_data["filter_dict"][c0][0]
    indx_c1 = pdata.diffsky_data["filter_dict"][c1][0]
    c0_pred = pdata.diffsky_data["obs_mags"][:, indx_c0][msk_sample_pred]
    c1_pred = pdata.diffsky_data["obs_mags"][:, indx_c1][msk_sample_pred]
    color_pred = c0_pred - c1_pred
    ax.hist(
        color_pred,
        bins=target_bins,
        label=r"${\rm Diffsky}$",
        density=True,
        alpha=0.7,
        weights=w_pred,
    )

    ax.set_title(f"{m1_label} = {m1_bin}; z={z_bin}")
    ax.legend()

    prefix = f"{model_nickname}_diffsky_vs_cosmos_"
    color_label = f"{c0_label}-{c1_label}_PDF_"
    z_m_label = f"z={z_bin:.1f}_{m1_label}={m1_bin:.1f}"
    bn_out = prefix + color_label + z_m_label + ".png"
    fn_out = os.path.join(drn_out, bn_out)
    fig.savefig(
        fn_out, bbox_extra_artists=[xlabel, ylabel], bbox_inches="tight", dpi=200
    )
    plt.close()

    return fig


def make_color_mag_diagnostic_plots(
    *,
    param_collection,
    model_nickname,
    drn_out,
    z_bins=(0.6, 1.0, 2.0),
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
    m0=c20.HSC_MAG_NAMES[0],
    m1=c20.HSC_MAG_NAMES[2],
    m2=c20.UVISTA_MAG_NAMES[1],
    z_magi_pairs=Z_MAGI_PAIRS,
    color_pairs=COLOR_PAIRS,
    pdata=None,
    skip_param_check=False,
):
    """Generate standard set of diagnostic plots

    Parameters
    ----------
    param_collection : namedtuple
        Example: diffsky_param_wrappers.DEFAULT_PARAM_COLLECTION

    model_nickname : string
        Nickname of the model being tested.
        The model_nickname will be part of the output filename of each plot.

    drn_out : string
        Output directory to store plots

    """
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG

    if pdata is None:
        pdata = get_plotting_data(
            seed=0,
            **param_collection._asdict(),
            cosmo_params=cosmo_params,
            fb=fb,
            skip_param_check=skip_param_check,
        )

    for z_bin in z_bins:
        plot_app_mag_func(
            pdata=pdata,
            z_bin=z_bin,
            drn_out=os.path.join(drn_out, "app_mag_funcs"),
            model_nickname=model_nickname,
            m0=m0,
            m1=m1,
            m2=m2,
        )

    for z_bin, m_i in z_magi_pairs:
        for c0, c1 in color_pairs:
            plot_color_pdf(
                pdata=pdata,
                z_bin=z_bin,
                m1_bin=m_i,
                drn_out=os.path.join(drn_out, "color_pdfs"),
                model_nickname=model_nickname,
                m1=m1,
                c0=c0,
                c1=c1,
            )
