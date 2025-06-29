""" """

import h5py
import numpy as np
from diffmah import DEFAULT_MAH_PARAMS, logmh_at_t_obs
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstarpop import mc_diffstar_sfh_galpop
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstarpop.param_utils import mc_select_diffstar_params
from dsps.cosmology import flat_wcdm
from jax import random as jran

from ...fake_sats import halo_boundary_functions as hbf
from ...fake_sats import nfw_config_space as nfwcs
from ...utils.sfh_utils import get_logsm_logssfr_at_t_obs
from . import load_lc_cf

LC_CF_BNPAT = "lc_cores-{0}.{1}.diffsky_data.hdf5"
LC_MOCK_BNPAT = LC_CF_BNPAT.replace("diffsky_data", "diffsky_gals")


def write_lc_sfh_mock_to_disk(fnout, lc_data, diffsky_data):
    with h5py.File(fnout, "w") as hdf_out:

        # Write diffmah params
        for key in DEFAULT_MAH_PARAMS._fields:
            hdf_out[key] = diffsky_data[key]
        hdf_out["has_diffmah_fit"] = diffsky_data["has_diffmah_fit"]
        hdf_out["logmp0"] = diffsky_data["logmp0"]
        hdf_out["logmp_obs"] = diffsky_data["logmp_obs"]

        # Write diffstar params
        for key in DEFAULT_DIFFSTAR_PARAMS.ms_params._fields:
            hdf_out[key] = diffsky_data[key]
        for key in DEFAULT_DIFFSTAR_PARAMS.q_params._fields:
            hdf_out[key] = diffsky_data[key]
        hdf_out["logsm_obs"] = diffsky_data["logsm_obs"]
        hdf_out["logssfr_obs"] = diffsky_data["logssfr_obs"]

        hdf_out["z_true"] = lc_data["z_true"]
        hdf_out["ra"] = lc_data["phi"]
        hdf_out["dec"] = np.pi / 2.0 - lc_data["theta"]
        hdf_out["snapnum"] = lc_data["snapnum"]

        lc_data_keys_out = ("core_tag", "x", "y", "z", "top_host_idx")
        for key in lc_data_keys_out:
            hdf_out[key] = lc_data[key]

        diffsky_data_keys_out = ("host_x", "host_y", "host_z", "logmp_obs_host")
        for key in diffsky_data_keys_out:
            hdf_out[key] = diffsky_data[key]


def add_sfh_quantities_to_mock(sim_info, lc_data, diffsky_data, ran_key):
    lc_data["t_obs"] = flat_wcdm.age_at_z(lc_data["z_true"], *sim_info.cosmo_params)

    mah_params, msk_has_diffmah_fit = load_lc_cf.get_imputed_mah_params(
        ran_key, diffsky_data, lc_data, sim_info.lgt0
    )
    for pname, pval in zip(mah_params._fields, mah_params):
        diffsky_data[pname] = pval
    diffsky_data["has_diffmah_fit"] = msk_has_diffmah_fit

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

    t_table = np.linspace(0.1, 10**sim_info.lgt0, 100)

    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params,
        logmp0,
        upids,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
    )

    _res = mc_diffstar_sfh_galpop(*args)
    sfh_params_ms, sfh_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

    sfh = np.where(mc_is_q.reshape((-1, 1)), sfh_q, sfh_ms)
    sfh_params = mc_select_diffstar_params(sfh_params_q, sfh_params_ms, mc_is_q)

    for key in sfh_params.ms_params._fields:
        diffsky_data[key] = getattr(sfh_params.ms_params, key)
    for key in sfh_params.q_params._fields:
        diffsky_data[key] = getattr(sfh_params.q_params, key)

    logsm_obs, logssfr_obs = get_logsm_logssfr_at_t_obs(lc_data["t_obs"], t_table, sfh)
    diffsky_data["logsm_obs"] = logsm_obs
    diffsky_data["logssfr_obs"] = logssfr_obs

    return lc_data, diffsky_data


def reposition_satellites(sim_info, lc_data, diffsky_data, ran_key, fixed_conc=5.0):

    pos = np.array((lc_data["x"], lc_data["y"], lc_data["z"])).T
    host_pos = [lc_data[key][lc_data["top_host_idx"]] for key in ("x", "y", "z")]
    host_pos = np.array(host_pos).T
    host_logmp_obs = diffsky_data["logmp_obs"][lc_data["top_host_idx"]]

    diffsky_data["logmp_obs_host"] = host_logmp_obs
    diffsky_data["x_host"] = host_pos[:, 0]
    diffsky_data["y_host"] = host_pos[:, 1]
    diffsky_data["z_host"] = host_pos[:, 2]

    args = (10**host_logmp_obs, sim_info.cosmo_params, lc_data["z_true"], "200m")
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
    msk_cen = np.reshape(lc_data["central"] == 1, (n_cores, 3))
    new_pos = np.where(msk_cen, pos, new_pos)
    lc_data["x_nfw"] = new_pos[:, 0]
    lc_data["y_nfw"] = new_pos[:, 1]
    lc_data["z_nfw"] = new_pos[:, 2]

    return lc_data, diffsky_data


def get_patch_info_from_mock_basename(bn):
    stepnum, patchnum = bn[: bn.find(".diffsky_gals")].split("-")[1].split(".")
    stepnum = int(stepnum)
    patchnum = int(patchnum)
    return stepnum, patchnum
