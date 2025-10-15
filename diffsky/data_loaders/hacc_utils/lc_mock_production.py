# flake8: noqa: E402
"""Kernels used to produce the SFH mock lightcone"""

import jax

jax.config.update("jax_enable_x64", True)

import h5py
import numpy as np
from diffmah import DEFAULT_MAH_PARAMS, logmh_at_t_obs
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstar.diffstarpop import mc_diffstar_sfh_galpop
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstar.diffstarpop.param_utils import mc_select_diffstar_params
from dsps.cosmology import flat_wcdm
from jax import random as jran

from ...experimental import mc_diffsky_seds
from ...fake_sats import halo_boundary_functions as hbf
from ...fake_sats import nfw_config_space as nfwcs
from ...utils.sfh_utils import get_logsm_logssfr_at_t_obs
from . import lightcone_utils as hlu
from . import load_lc_cf

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
    "snapnum",
)

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
    "logsm_obs",
    "logssfr_obs",
    *TOP_HOST_SHAPE_KEYS,
    *DEFAULT_MAH_PARAMS._fields,
    *DEFAULT_DIFFSTAR_PARAMS._fields,
)

PHOT_INFO_KEYS_OUT = (
    "uran_av",
    "uran_delta",
    "uran_funo",
    "delta_scatter_ms",
    "delta_scatter_q",
)


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

        gen = zip(phot_info["burst_params"]._fields, phot_info["burst_params"])
        for name, parr in gen:
            hdf_out["data"][name] = parr

        hdf_out["data"]["mc_sfh_type"] = phot_info["mc_sfh_type"]

        for name in PHOT_INFO_KEYS_OUT:
            hdf_out["data"][name] = phot_info[name]


def add_sfh_quantities_to_mock(sim_info, lc_data, diffsky_data, ran_key):
    lc_data["t_obs"] = flat_wcdm.age_at_z(
        lc_data["redshift_true"], *sim_info.cosmo_params
    )

    mah_params, msk_has_diffmah_fit = load_lc_cf.get_imputed_mah_params(
        ran_key, diffsky_data, lc_data, sim_info.lgt0
    )
    for pname, pval in zip(mah_params._fields, mah_params):
        diffsky_data[pname] = pval
    diffsky_data["has_diffmah_fit"] = msk_has_diffmah_fit
    diffsky_data["mah_params"] = mah_params

    logmp0 = logmh_at_t_obs(
        mah_params, np.zeros(mah_params.logm0.size) + 10**sim_info.lgt0, sim_info.lgt0
    )
    diffsky_data["logmp0"] = logmp0

    logmp_obs = logmh_at_t_obs(
        mah_params, np.zeros(mah_params.logm0.size) + lc_data["t_obs"], sim_info.lgt0
    )
    diffsky_data["logmp_obs"] = logmp_obs

    lgmu_infall = np.zeros_like(logmp0)
    logmhost_infall = np.copy(logmp0)
    gyr_since_infall = np.zeros_like(logmp0)
    upids = np.where(lc_data["central"] == 1, -1, 0)

    diffsky_data["t_table"] = np.linspace(0.1, 10**sim_info.lgt0, 100)

    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params,
        logmp0,
        upids,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        diffsky_data["t_table"],
    )

    _res = mc_diffstar_sfh_galpop(*args)
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
        lc_data["t_obs"], diffsky_data["t_table"], diffsky_data["sfh_table"]
    )
    diffsky_data["logsm_obs"] = logsm_obs
    diffsky_data["logssfr_obs"] = logssfr_obs

    logsm_obs_ms, logssfr_obs_ms = get_logsm_logssfr_at_t_obs(
        lc_data["t_obs"], diffsky_data["t_table"], diffsky_data["sfh_table_ms"]
    )
    diffsky_data["logsm_obs_ms"] = logsm_obs_ms
    diffsky_data["logssfr_obs_ms"] = logssfr_obs_ms

    logsm_obs_q, logssfr_obs_q = get_logsm_logssfr_at_t_obs(
        lc_data["t_obs"], diffsky_data["t_table"], diffsky_data["sfh_table_q"]
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

    ran_key, sed_key = jran.split(ran_key, 2)
    args = (
        sed_key,
        lc_data["redshift_true"],
        lc_data["t_obs"],
        diffsky_data["mah_params"],
        diffsky_data["logmp0"],
        diffsky_data["t_table"],
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


def add_morphology_quantities_to_mock(
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
    pass


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
