# flake8: noqa: E402
"""Kernels used to produce the SFH mock lightcone"""

import os
from collections import namedtuple

import jax
from dsps.data_loaders import load_transmission_curve
from dsps.data_loaders.load_filter_data import TransmissionCurve
from dsps.data_loaders.load_ssp_data import SSPData

from ...param_utils import diffsky_param_wrapper as dpw
from .. import io_utils as iou

jax.config.update("jax_enable_x64", True)
import h5py
import numpy as np
from diffmah import DEFAULT_MAH_PARAMS, logmh_at_t_obs
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstar.defaults import T_TABLE_MIN
from diffstar.diffstarpop import mc_diffstar_sfh_galpop
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstar.diffstarpop.param_utils import mc_select_diffstar_params
from dsps.cosmology import flat_wcdm
from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ...dustpop.tw_dust import DEFAULT_DUST_PARAMS
from ...ellipsoidal_shapes import bulge_shapes, disk_shapes, ellipse_proj_kernels
from ...experimental import mc_diffsky_disk_bulge_knot_seds as mc_dbk_sed
from ...experimental import mc_diffsky_seds, mc_phot_repro
from ...experimental.black_hole_modeling import black_hole_mass as bhm
from ...experimental.black_hole_modeling.black_hole_accretion_rate import (
    monte_carlo_bh_acc_rate,
)
from ...experimental.black_hole_modeling.utils import approximate_ssfr_percentile
from ...experimental.disk_bulge_modeling import disk_bulge_kernels as dbk
from ...experimental.size_modeling import disk_bulge_sizes as dbs
from ...fake_sats import halo_boundary_functions as hbf
from ...fake_sats import nfw_config_space as nfwcs
from ...utils.sfh_utils import get_logsm_logssfr_at_t_obs
from .. import load_flat_hdf5
from . import lightcone_utils as hlu
from . import load_lc_cf

N_T_TABLE = 100

LC_CF_BNPAT = "lc_cores-{0}.{1}.diffsky_data.hdf5"
LC_MOCK_BNPAT = LC_CF_BNPAT.replace("diffsky_data", "diffsky_gals")

shapes_1 = [f"infall_fof_halo_eigS1{x}" for x in ("X", "Y", "Z")]
shapes_2 = [f"infall_fof_halo_eigS2{x}" for x in ("X", "Y", "Z")]
shapes_3 = [f"infall_fof_halo_eigS3{x}" for x in ("X", "Y", "Z")]
SHAPE_KEYS = (*shapes_1, *shapes_2, *shapes_3)
TOP_HOST_SHAPE_KEYS = ["top_host_" + key for key in SHAPE_KEYS]

LC_DATA_KEYS_OUT = (
    "core_tag",
    "x",
    "y",
    "z",
    "x_nfw",
    "y_nfw",
    "z_nfw",
    "top_host_idx",
    "central",
    "ra_nfw",
    "dec_nfw",
    "redshift_true",
    "stepnum",
    "lc_patch",
)

SIZE_KEYS = ("r50_disk", "r50_bulge", "zscore_r50_disk", "zscore_r50_bulge")
_ORIEN_PATS = (
    "b_over_a_{}",
    "c_over_a_{}",
    "beta_{}",
    "alpha_{}",
    "ellipticity_{}",
    "psi_{}",
    "e_beta_x_{}",
    "e_beta_y_{}",
    "e_alpha_x_{}",
    "e_alpha_y_{}",
)
ORIENTATION_KEYS = [pat.format("disk") for pat in _ORIEN_PATS]
ORIENTATION_KEYS.extend([pat.format("bulge") for pat in _ORIEN_PATS])

DIFFSKY_DATA_KEYS_OUT = (
    "x_host",
    "y_host",
    "z_host",
    "vx",
    "vy",
    "vz",
    "logmp_obs_host",
    "has_diffmah_fit",
    "logmp0",
    "logmp_obs",
    *TOP_HOST_SHAPE_KEYS,
    *DEFAULT_MAH_PARAMS._fields,
    *SIZE_KEYS,
    *ORIENTATION_KEYS,
)

PHOT_INFO_KEYS_OUT = (
    *DEFAULT_DIFFSTAR_PARAMS._fields,
    "logsm_obs",
    "logssfr_obs",
    "uran_av",
    "uran_delta",
    "uran_funo",
    "delta_scatter_ms",
    "delta_scatter_q",
)

MORPH_KEYS_OUT = ("bulge_to_total", *dbk.DEFAULT_FBULGE_PARAMS._fields)
BLACK_HOLE_KEYS_OUT = (
    "black_hole_mass",
    "black_hole_eddington_ratio",
    "black_hole_accretion_rate",
)


interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))


BNPAT_TCURVES = "diffsky_{0}_transmission_curves.hdf5"
BNPAT_SSP_DATA = "diffsky_{0}_ssp_data.hdf5"
BNPAT_PARAM_COLLECTION = "diffsky_{0}_param_collection.hdf5"


def write_diffsky_ssp_data_to_disk(drn_out, mock_version_name, ssp_data):
    """"""
    bn_ssp_data = BNPAT_SSP_DATA.format(mock_version_name)
    with h5py.File(os.path.join(drn_out, bn_ssp_data), "w") as hdf_out:
        for name, arr in zip(ssp_data._fields, ssp_data):
            hdf_out[name] = arr


def load_diffsky_ssp_data(drn_mock, mock_version_name):
    bn_ssp_data = BNPAT_SSP_DATA.format(mock_version_name)
    fn_ssp_data = os.path.join(drn_mock, bn_ssp_data)
    ssp_data_dict = load_flat_hdf5(fn_ssp_data)

    ssp_data = SSPData(*[ssp_data_dict[key] for key in SSPData._fields])
    return ssp_data


def write_diffsky_tcurves_to_disk(
    drn_out, mock_version_name, tcurves, filter_nicknames
):
    """"""
    bn_tcurves = BNPAT_TCURVES.format(mock_version_name)
    with h5py.File(os.path.join(drn_out, bn_tcurves), "w") as hdf_out:
        for tcurve, nickname in zip(tcurves, filter_nicknames):
            tcurve_group = hdf_out.require_group(nickname)
            tcurve_group["wave"] = tcurve.wave
            tcurve_group["transmission"] = tcurve.transmission

        hdf_out.attrs["_fields"] = np.array(filter_nicknames, dtype="S")


def load_diffsky_tcurves(drn_mock, mock_version_name):
    """"""
    bn_tcurves = BNPAT_TCURVES.format(mock_version_name)
    fn = os.path.join(drn_mock, bn_tcurves)
    with h5py.File(fn, "r") as hdf:
        filter_nicknames = [
            f.decode() if isinstance(f, bytes) else f for f in hdf.attrs["_fields"]
        ]
        tcurves = []
        for nickname in filter_nicknames:
            tcurve = TransmissionCurve(
                hdf[nickname]["wave"][:], hdf[nickname]["transmission"][:]
            )
            tcurves.append(tcurve)

        TCurves = namedtuple("TCurves", filter_nicknames)
        tcurves = TCurves(*tcurves)
    return tcurves


def load_diffsky_param_collection(drn_mock, mock_version_name):
    """"""
    bn = BNPAT_PARAM_COLLECTION.format(mock_version_name)
    fn = os.path.join(drn_mock, bn)
    flat_diffsky_params = iou.load_namedtuple_from_hdf5(fn)
    param_collection = dpw.get_param_collection_from_flat_array(flat_diffsky_params)
    return param_collection


def write_diffsky_param_collection(drn_mock, mock_version_name, param_collection):
    """"""
    bn = BNPAT_PARAM_COLLECTION.format(mock_version_name)
    fn_out = os.path.join(drn_mock, bn)
    flat_diffsky_params = dpw.unroll_param_collection_into_flat_array(*param_collection)
    DiffskyParams = namedtuple("DiffskyParams", dpw.get_flat_param_names())
    flat_diffsky_params = DiffskyParams(*flat_diffsky_params)

    iou.write_namedtuple_to_hdf5(flat_diffsky_params, fn_out)


def load_diffsky_sim_info(fn_mock):
    with h5py.File(fn_mock, "r") as hdf:
        sim_name = hdf["metadata/nbody_info"].attrs["sim_name"]
    sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim(sim_name)
    return sim_info


def get_dsps_transmission_curves(filter_nicknames, drn=None):
    bn_pat_list = [name + "*" for name in filter_nicknames]
    TCurves = namedtuple("TCurves", filter_nicknames)
    tcurves = TCurves(
        *[load_transmission_curve(bn_pat=bn_pat, drn=drn) for bn_pat in bn_pat_list]
    )
    return tcurves


def write_diffsky_t_table(drn_mock, mock_version_name, sim_info):
    t_table = np.linspace(T_TABLE_MIN, 10**sim_info.lgt0, N_T_TABLE)

    bn_t_table = f"diffsky_{mock_version_name}_t_table.hdf5"
    fn_t_table = os.path.join(drn_mock, bn_t_table)
    with h5py.File(fn_t_table, "w") as hdf_out:
        hdf_out["t_table"] = t_table


def load_diffsky_t_table(drn_mock, mock_version_name):
    bn_t_table = f"diffsky_{mock_version_name}_t_table.hdf5"
    fn_t_table = os.path.join(drn_mock, bn_t_table)
    with h5py.File(fn_t_table, "r") as hdf:
        t_table = hdf["t_table"][:]
    return t_table


def load_diffsky_z_phot_table(fn_mock):
    with h5py.File(fn_mock, "r") as hdf:
        z_phot_table = hdf["metadata/z_phot_table"][:]
    return z_phot_table


def write_lc_sfh_mock_to_disk(fnout, lc_data, diffsky_data):
    with h5py.File(fnout, "w") as hdf_out:

        hdf_out.require_group("data")

        ra, dec = hlu._get_lon_lat_from_theta_phi(lc_data["theta"], lc_data["phi"])
        hdf_out["data/ra"] = ra
        hdf_out["data/dec"] = dec

        for key in LC_DATA_KEYS_OUT:
            key_out = "data/" + key
            hdf_out[key_out] = lc_data[key]

        for key in DIFFSKY_DATA_KEYS_OUT:
            key_out = "data/" + key
            hdf_out[key_out] = diffsky_data[key]


def write_lc_sed_mock_to_disk(
    fnout, phot_info, lc_data, diffsky_data, filter_nicknames
):
    write_lc_sfh_mock_to_disk(fnout, lc_data, diffsky_data)

    with h5py.File(fnout, "a") as hdf_out:
        for iband, name in enumerate(filter_nicknames):
            hdf_out["data"][name] = phot_info["obs_mags"][:, iband]

        for burst_pname in DEFAULT_BURST_PARAMS._fields:
            hdf_out["data"][burst_pname] = phot_info[burst_pname]

        for dust_pname in DEFAULT_DUST_PARAMS._fields:
            hdf_out["data"][dust_pname] = phot_info[dust_pname]

        hdf_out["data"]["mc_sfh_type"] = phot_info["mc_sfh_type"]

        for name in PHOT_INFO_KEYS_OUT:
            hdf_out["data"][name] = phot_info[name]

        for name in MORPH_KEYS_OUT:
            hdf_out["data"][name] = diffsky_data[name]

        for name in BLACK_HOLE_KEYS_OUT:
            hdf_out["data"][name] = diffsky_data[name]


def write_lc_dbk_sed_mock_to_disk(
    fnout, phot_info, lc_data, diffsky_data, filter_nicknames
):
    write_lc_sed_mock_to_disk(fnout, phot_info, lc_data, diffsky_data, filter_nicknames)

    with h5py.File(fnout, "a") as hdf_out:
        for iband, name in enumerate(filter_nicknames):
            hdf_out["data"][name + "_bulge"] = phot_info["obs_mags_bulge"][:, iband]
            hdf_out["data"][name + "_disk"] = phot_info["obs_mags_disk"][:, iband]
            hdf_out["data"][name + "_knots"] = phot_info["obs_mags_knots"][:, iband]
        hdf_out["data"]["fknot"] = phot_info["fknot"]


def add_sfh_quantities_to_mock(sim_info, lc_data, diffsky_data, ran_key):
    mah_key, sfh_key = jran.split(ran_key, 2)

    diffsky_data["t_obs"] = flat_wcdm.age_at_z(
        lc_data["redshift_true"], *sim_info.cosmo_params
    )

    mah_params, msk_has_diffmah_fit = load_lc_cf.get_imputed_mah_params(
        mah_key, diffsky_data, sim_info.lgt0
    )
    for pname, pval in zip(mah_params._fields, mah_params):
        diffsky_data[pname] = pval
    diffsky_data["has_diffmah_fit"] = msk_has_diffmah_fit

    logmp0 = logmh_at_t_obs(
        mah_params, np.zeros(mah_params.logm0.size) + 10**sim_info.lgt0, sim_info.lgt0
    )
    diffsky_data["logmp0"] = logmp0

    logmp_obs = logmh_at_t_obs(
        mah_params,
        np.zeros(mah_params.logm0.size) + diffsky_data["t_obs"],
        sim_info.lgt0,
    )
    diffsky_data["logmp_obs"] = logmp_obs

    lgmu_infall = np.zeros_like(logmp0)
    logmhost_infall = np.copy(logmp0)
    gyr_since_infall = np.zeros_like(logmp0)
    upids = np.where(lc_data["central"] == 1, -1, 0)

    t_table = np.linspace(T_TABLE_MIN, 10**sim_info.lgt0, N_T_TABLE)

    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params,
        logmp0,
        upids,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        sfh_key,
        t_table,
    )

    _res = mc_diffstar_sfh_galpop(*args, lgt0=sim_info.lgt0, fb=sim_info.fb)
    sfh_params_ms, sfh_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

    diffsky_data["sfh_table"] = np.where(mc_is_q.reshape((-1, 1)), sfh_q, sfh_ms)
    sfh_params = mc_select_diffstar_params(sfh_params_q, sfh_params_ms, mc_is_q)

    diffsky_data["sfh_table_ms"] = sfh_ms
    diffsky_data["sfh_table_q"] = sfh_q

    diffsky_data["mc_is_q"] = mc_is_q
    # Need to add SFH table after MC selection

    for key in sfh_params._fields:
        diffsky_data[key] = getattr(sfh_params, key)

    logsm_obs, logssfr_obs = get_logsm_logssfr_at_t_obs(
        diffsky_data["t_obs"], t_table, diffsky_data["sfh_table"]
    )
    diffsky_data["logsm_obs"] = logsm_obs
    diffsky_data["logssfr_obs"] = logssfr_obs

    logsm_obs_ms, logssfr_obs_ms = get_logsm_logssfr_at_t_obs(
        diffsky_data["t_obs"], t_table, diffsky_data["sfh_table_ms"]
    )
    diffsky_data["logsm_obs_ms"] = logsm_obs_ms
    diffsky_data["logssfr_obs_ms"] = logssfr_obs_ms

    logsm_obs_q, logssfr_obs_q = get_logsm_logssfr_at_t_obs(
        diffsky_data["t_obs"], t_table, diffsky_data["sfh_table_q"]
    )
    diffsky_data["logsm_obs_q"] = logsm_obs_q
    diffsky_data["logssfr_obs_q"] = logssfr_obs_q

    return lc_data, diffsky_data


def add_sed_quantities_to_mock(
    sim_info,
    lc_data,
    diffsky_data,
    ssp_data,
    param_collection,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    ran_key,
):
    (
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
    ) = param_collection

    n_z_table, n_bands, n_met, n_age = precomputed_ssp_mag_table.shape

    ran_key, sfh_key = jran.split(ran_key, 2)
    lc_data, diffsky_data = add_sfh_quantities_to_mock(
        sim_info, lc_data, diffsky_data, sfh_key
    )
    n_gals = diffsky_data["logsm_obs"].size

    t_table = np.linspace(T_TABLE_MIN, 10**sim_info.lgt0, N_T_TABLE)

    ran_key, sed_key = jran.split(ran_key, 2)
    args = (
        sed_key,
        lc_data["redshift_true"],
        diffsky_data["t_obs"],
        diffsky_data["mah_params"],
        diffsky_data["logmp0"],
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        sim_info.cosmo_params,
    )
    phot_info = mc_diffsky_seds._mc_diffsky_phot_kern(*args)
    return phot_info, lc_data, diffsky_data


def add_diffmah_properties_to_mock(diffsky_data, redshift_true, sim_info, ran_key):
    diffsky_data["t_obs"] = flat_wcdm.age_at_z(redshift_true, *sim_info.cosmo_params)

    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )

    mah_params, msk_has_diffmah_fit = load_lc_cf.get_imputed_mah_params(
        ran_key, diffsky_data, sim_info.lgt0
    )
    for pname, pval in zip(mah_params._fields, mah_params):
        diffsky_data[pname] = pval
    diffsky_data["has_diffmah_fit"] = msk_has_diffmah_fit

    diffsky_data["logmp0"] = logmh_at_t_obs(
        mah_params,
        np.zeros(mah_params.logm0.size) + 10**sim_info.lgt0,
        sim_info.lgt0,
    )

    diffsky_data["logmp_obs"] = logmh_at_t_obs(
        mah_params,
        np.zeros(mah_params.logm0.size) + diffsky_data["t_obs"],
        sim_info.lgt0,
    )
    return diffsky_data


def add_dbk_phot_quantities_to_mock(
    sim_info,
    lc_data,
    diffsky_data,
    ssp_data,
    param_collection,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    ran_key,
):
    ran_key, mah_key = jran.split(ran_key, 2)
    diffsky_data = add_diffmah_properties_to_mock(
        diffsky_data, lc_data["redshift_true"], sim_info, mah_key
    )

    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )

    dbk_phot_info = mc_phot_repro.mc_dbk_phot(
        ran_key,
        lc_data["redshift_true"],
        diffsky_data["t_obs"],
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        *param_collection,
        sim_info.cosmo_params,
        sim_info.fb,
    )

    # Discard columns storing non-tabular data
    dbk_phot_info = dbk_phot_info._asdict()
    dbk_phot_info.pop("burstiness")
    dbk_phot_info.pop("t_table")

    return dbk_phot_info, lc_data, diffsky_data


def add_dbk_sed_quantities_to_mock(
    sim_info,
    lc_data,
    diffsky_data,
    ssp_data,
    param_collection,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    ran_key,
):

    ran_key, mah_key = jran.split(ran_key, 2)
    diffsky_data = add_diffmah_properties_to_mock(
        diffsky_data, lc_data["redshift_true"], sim_info, mah_key
    )

    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )

    t_table = np.linspace(T_TABLE_MIN, 10**sim_info.lgt0, N_T_TABLE)

    ran_key, phot_key = jran.split(ran_key, 2)
    args = (
        phot_key,
        lc_data["redshift_true"],
        diffsky_data["t_obs"],
        mah_params,
        diffsky_data["logmp0"],
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        *param_collection,
        sim_info.cosmo_params,
        sim_info.fb,
    )
    phot_info = mc_dbk_sed._mc_diffsky_disk_bulge_knot_phot_kern(*args)

    return phot_info, lc_data, diffsky_data


def add_morphology_quantities_to_diffsky_data(
    sim_info, phot_info, lc_data, diffsky_data, morph_key
):
    for pname in dbk.DEFAULT_FBULGE_PARAMS._fields:
        diffsky_data[pname] = phot_info[pname]

    t_table = np.linspace(T_TABLE_MIN, 10**sim_info.lgt0, N_T_TABLE)

    diffsky_data["bulge_to_total"] = interp_vmap(
        diffsky_data["t_obs"],
        t_table,
        phot_info["bulge_to_total_history"],
    )

    morph_key, disk_size_key, bulge_size_key = jran.split(morph_key, 3)
    r50_disk, zscore_disk = dbs.mc_r50_disk_size(
        10 ** phot_info["logsm_obs"], lc_data["redshift_true"], disk_size_key
    )
    r50_bulge, zscore_bulge = dbs.mc_r50_bulge_size(
        10 ** phot_info["logsm_obs"], lc_data["redshift_true"], bulge_size_key
    )

    diffsky_data["r50_disk"] = r50_disk
    diffsky_data["r50_bulge"] = r50_bulge
    diffsky_data["zscore_r50_disk"] = zscore_disk
    diffsky_data["zscore_r50_bulge"] = zscore_bulge

    morph_key, disk_shape_key, bulge_shape_key = jran.split(morph_key, 3)
    n = diffsky_data["r50_disk"].size
    disk_axis_ratios = disk_shapes.sample_disk_axis_ratios(disk_shape_key, n)
    bulge_axis_ratios = bulge_shapes.sample_bulge_axis_ratios(bulge_shape_key, n)

    diffsky_data["b_over_a_disk"] = disk_axis_ratios.b_over_a
    diffsky_data["c_over_a_disk"] = disk_axis_ratios.c_over_a

    diffsky_data["b_over_a_bulge"] = bulge_axis_ratios.b_over_a
    diffsky_data["c_over_a_bulge"] = bulge_axis_ratios.c_over_a

    morph_key, disk_orientation_key, bulge_orientation_key = jran.split(morph_key, 3)
    ellipse2d_disk = ellipse_proj_kernels.mc_ellipsoid_params(
        diffsky_data["r50_disk"],
        diffsky_data["b_over_a_disk"],
        diffsky_data["c_over_a_disk"],
        disk_orientation_key,
    )
    ellipse2d_bulge = ellipse_proj_kernels.mc_ellipsoid_params(
        diffsky_data["r50_bulge"],
        diffsky_data["b_over_a_bulge"],
        diffsky_data["c_over_a_bulge"],
        bulge_orientation_key,
    )

    diffsky_data["beta_disk"] = ellipse2d_disk.beta
    diffsky_data["alpha_disk"] = ellipse2d_disk.alpha
    diffsky_data["ellipticity_disk"] = ellipse2d_disk.ellipticity
    diffsky_data["psi_disk"] = ellipse2d_disk.psi

    diffsky_data["e_beta_x_disk"] = ellipse2d_disk.e_beta[:, 0]
    diffsky_data["e_beta_y_disk"] = ellipse2d_disk.e_beta[:, 1]
    diffsky_data["e_alpha_x_disk"] = ellipse2d_disk.e_alpha[:, 0]
    diffsky_data["e_alpha_y_disk"] = ellipse2d_disk.e_alpha[:, 1]

    diffsky_data["beta_bulge"] = ellipse2d_bulge.beta
    diffsky_data["alpha_bulge"] = ellipse2d_bulge.alpha
    diffsky_data["ellipticity_bulge"] = ellipse2d_bulge.ellipticity
    diffsky_data["psi_bulge"] = ellipse2d_bulge.psi

    diffsky_data["e_beta_x_bulge"] = ellipse2d_bulge.e_beta[:, 0]
    diffsky_data["e_beta_y_bulge"] = ellipse2d_bulge.e_beta[:, 1]
    diffsky_data["e_alpha_x_bulge"] = ellipse2d_bulge.e_alpha[:, 0]
    diffsky_data["e_alpha_y_bulge"] = ellipse2d_bulge.e_alpha[:, 1]

    return diffsky_data


def add_black_hole_quantities_to_diffsky_data(lc_data, diffsky_data, phot_info):
    bulge_mass = diffsky_data["bulge_to_total"] * 10 ** phot_info["logsm_obs"]
    diffsky_data["black_hole_mass"] = bhm.bh_mass_from_bulge_mass(bulge_mass)

    p_ssfr = approximate_ssfr_percentile(phot_info["logssfr_obs"])
    z = lc_data["redshift_true"].mean()
    _res = monte_carlo_bh_acc_rate(z, diffsky_data["black_hole_mass"], p_ssfr)
    diffsky_data["black_hole_eddington_ratio"] = _res[0]
    diffsky_data["black_hole_accretion_rate"] = _res[1]

    return diffsky_data


def reposition_satellites(sim_info, lc_data, diffsky_data, ran_key, fixed_conc=5.0):

    pos = np.array((lc_data["x"], lc_data["y"], lc_data["z"])).T
    host_pos = [lc_data[key][lc_data["top_host_idx"]] for key in ("x", "y", "z")]
    host_pos = np.array(host_pos).T
    host_logmp_obs = diffsky_data["logmp_obs"][lc_data["top_host_idx"]]

    diffsky_data["logmp_obs_host"] = host_logmp_obs
    diffsky_data["x_host"] = host_pos[:, 0]
    diffsky_data["y_host"] = host_pos[:, 1]
    diffsky_data["z_host"] = host_pos[:, 2]

    args = (10**host_logmp_obs, sim_info.cosmo_params, lc_data["redshift_true"], "200m")
    host_radius_mpc = hbf.halo_mass_to_halo_radius(*args) / 1000.0

    n_cores = host_logmp_obs.shape[0]
    axis_key, nfw_key = jran.split(ran_key, 2)
    major_axes = jran.uniform(axis_key, minval=-1, maxval=1, shape=(n_cores, 3))
    b_to_a = np.ones(n_cores)
    c_to_a = np.ones(n_cores)
    conc = np.zeros(n_cores) + fixed_conc

    args = (nfw_key, host_radius_mpc, conc, major_axes, b_to_a, c_to_a)
    host_centric_pos = nfwcs.mc_ellipsoidal_positions(*args)

    new_pos = host_centric_pos + host_pos
    msk_cen = np.reshape(lc_data["central"] == 1, (n_cores, 1))
    new_pos = np.where(msk_cen, pos, new_pos)
    lc_data["x_nfw"] = new_pos[:, 0]
    lc_data["y_nfw"] = new_pos[:, 1]
    lc_data["z_nfw"] = new_pos[:, 2]

    ra, dec = hlu.get_ra_dec(lc_data["x_nfw"], lc_data["y_nfw"], lc_data["z_nfw"])
    lc_data["ra_nfw"] = ra
    lc_data["dec_nfw"] = dec

    return lc_data, diffsky_data


def get_patch_info_from_mock_basename(bn):
    stepnum, patchnum = bn[: bn.find(".diffsky_gals")].split("-")[1].split(".")
    stepnum = int(stepnum)
    patchnum = int(patchnum)
    return stepnum, patchnum


def concatenate_batched_phot_data(phot_batches):

    phot_info = dict()
    for key in phot_batches[0][0].keys():
        try:
            phot_info[key] = np.concatenate([x[0][key] for x in phot_batches])
        except ValueError:
            raise ValueError(f"Unable to concatenate phot_info['{key}']")

    lc_data = dict()
    for key in phot_batches[0][1].keys():
        try:
            lc_data[key] = np.concatenate([x[1][key] for x in phot_batches])
        except ValueError:
            raise ValueError(f"Unable to concatenate lc_data['{key}']")

    diffsky_data = dict()
    for key in phot_batches[0][2].keys():
        try:
            diffsky_data[key] = np.concatenate([x[2][key] for x in phot_batches])
        except ValueError:
            raise ValueError(f"Unable to concatenate diffsky_data['{key}']")

    return phot_info, lc_data, diffsky_data
