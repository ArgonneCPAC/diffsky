""" """

import os
from collections import namedtuple

import numpy as np
from diffmah import logmh_at_t_obs
from jax import random as jran

from ...experimental import lc_utils
from ...experimental import mc_lightcone_halos as mclh
from . import haccsims
from . import lightcone_utils as hlu
from . import load_lc_cf as llcf

LCPKEYS = ("z_lo", "z_hi", "ra_lo", "ra_hi", "dec_lo", "dec_hi", "sky_area_degsq")
LCPatchInfo = namedtuple("LCPatchInfo", LCPKEYS)

STDVEL_COSMIC = 500.0


def load_lc_diffsky_patch_data(
    fn_lc_cores, sim_name, ran_key, lgmp_min, lgmp_max, *, downsample_factor=1.0
):

    sim_info = llcf.get_diffsky_info_from_hacc_sim(sim_name)

    bname_lc_cores = os.path.basename(fn_lc_cores)
    stepnum, lc_patch = hlu.get_stepnum_and_skypatch_from_lc_bname(bname_lc_cores)

    _res = hlu.read_hacc_lc_patch_decomposition(sim_name)
    patch_decomposition, sky_frac, solid_angles = _res
    sky_area_degsq = solid_angles[lc_patch]
    sky_area_degsq = sky_area_degsq / downsample_factor

    a_min, a_max = hlu.get_a_range_of_lc_cores_file(bname_lc_cores, sim_name)

    z_min = 1 / a_max - 1
    z_max = 1 / a_min - 1
    ran_key, mah_key = jran.split(ran_key, 2)
    args = (
        mah_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
    )
    diffsky_data = mclh.mc_lightcone_host_halo_diffmah(
        *args, logmp_cutoff=11.0, lgmp_max=lgmp_max
    )
    diffsky_data["redshift_true"] = diffsky_data["z_obs"]
    del diffsky_data["z_obs"]

    n_gals = len(diffsky_data["redshift_true"])

    theta_lo, theta_hi, phi_lo, phi_hi = [
        patch_decomposition[lc_patch, i] for i in range(1, 5)
    ]
    ra_lo, ra_hi, dec_lo, dec_hi = hlu._get_ra_dec_bounds(
        theta_lo, theta_hi, phi_lo, phi_hi
    )
    ran_key, ra_dec_key = jran.split(ran_key, 2)
    ra, dec = lc_utils.mc_lightcone_random_ra_dec(
        ra_dec_key, n_gals, ra_lo, ra_hi, dec_lo, dec_hi
    )
    diffsky_data["ra"] = ra
    diffsky_data["dec"] = dec
    theta, phi = hlu.get_theta_phi_from_ra_dec(ra, dec)
    diffsky_data["theta"] = theta
    diffsky_data["phi"] = phi

    diffsky_data["top_host_idx"] = np.arange(n_gals).astype(int)

    for key in diffsky_data["mah_params"]._fields:
        diffsky_data[key] = getattr(diffsky_data["mah_params"], key)

    diffsky_data["logmp_obs"] = logmh_at_t_obs(
        diffsky_data["mah_params"], diffsky_data["t_obs"], sim_info.lgt0
    )
    diffsky_data["logmp_obs_host"] = diffsky_data["logmp_obs"][
        diffsky_data["top_host_idx"]
    ]

    diffsky_data.pop("mah_params")

    eigh_pat = "top_host_infall_fof_halo_eigS{0}{1}"
    eigh_collector = []
    for n in (1, 2, 3):
        for s in ("X", "Y", "Z"):
            eigh_collector.append(eigh_pat.format(n, s))
    for key in (*eigh_collector,):
        diffsky_data[key] = np.zeros(n_gals) - 1.0

    x_mpc, y_mpc, z_mpc = hlu.get_xyz_mpc(
        diffsky_data["ra"],
        diffsky_data["dec"],
        diffsky_data["redshift_true"],
        sim_info.cosmo_params,
    )
    x_mpch = x_mpc * sim_info.cosmo_params.h
    y_mpch = y_mpc * sim_info.cosmo_params.h
    z_mpch = z_mpc * sim_info.cosmo_params.h
    diffsky_data["x"] = x_mpch
    diffsky_data["y"] = y_mpch
    diffsky_data["z"] = z_mpch
    diffsky_data["x_host"] = x_mpch
    diffsky_data["y_host"] = y_mpch
    diffsky_data["z_host"] = z_mpch

    ZZ = np.zeros(n_gals)

    ran_key, vel_key = jran.split(ran_key, 2)
    v_xyz = jran.normal(vel_key, shape=(n_gals, 3)) * STDVEL_COSMIC
    diffsky_data["vx"] = v_xyz[:, 0]
    diffsky_data["vy"] = v_xyz[:, 1]
    diffsky_data["vz"] = v_xyz[:, 2]

    diffsky_data["core_tag"] = -np.ones(len(diffsky_data["redshift_true"])).astype(int)
    diffsky_data["has_diffmah_fit"] = ZZ.astype(int) + 1
    diffsky_data["n_points_per_fit"] = ZZ.astype(int) + 10_000
    diffsky_data["loss"] = ZZ + 1e-5
    diffsky_data["central"] = ZZ.astype(int) + 1

    diffsky_data["stepnum"] = ZZ.astype(int) + stepnum

    lc_data = diffsky_data
    return lc_data, diffsky_data


def get_lc_patch_info_from_lc_cores(fn_lc_cores, sim_name):
    """Get lc_patch boundaries and sky area

    Parameters
    ----------
    fn_lc_cores : string

    sim_name : string

    Returns
    -------
    lc_patch_info : namedtuple
        lc_patch_info = z_lo, z_hi, ra_lo, ra_hi, dec_lo, dec_hi, sky_area_degsq

    """

    bname_lc_cores = os.path.basename(fn_lc_cores)
    stepnum, lc_patch = hlu.get_stepnum_and_skypatch_from_lc_bname(bname_lc_cores)

    _res = hlu.read_hacc_lc_patch_decomposition(sim_name)
    patch_decomposition, sky_frac, solid_angles = _res
    sky_area_degsq = solid_angles[lc_patch]

    theta_lo, theta_hi, phi_lo, phi_hi = patch_decomposition[lc_patch, 1:]
    ra_lo, dec_hi = hlu.get_ra_dec_from_theta_phi(theta_lo, phi_lo)
    ra_hi, dec_lo = hlu.get_ra_dec_from_theta_phi(theta_hi, phi_hi)

    z_table = haccsims.simulations[sim_name].redshifts
    steps = haccsims.simulations[sim_name].cosmotools_steps
    indx_stepnum = np.searchsorted(steps, stepnum)
    z_lo, z_hi = z_table[indx_stepnum + 1], z_table[indx_stepnum]

    return LCPatchInfo(z_lo, z_hi, ra_lo, ra_hi, dec_lo, dec_hi, sky_area_degsq)
