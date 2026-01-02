""" """

import os
from collections import namedtuple

import numpy as np
from diffmah import logmh_at_t_obs
from jax import random as jran

from ...experimental import mc_lightcone_halos as mclh
from . import haccsims, lc_mock
from . import lightcone_utils as hlu
from . import load_lc_cf as llcf

LCPKEYS = ("z_lo", "z_hi", "ra_lo", "ra_hi", "dec_lo", "dec_hi", "sky_area_degsq")
LCPatchInfo = namedtuple("LCPatchInfo", LCPKEYS)


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

    diffsky_data["top_host_idx"] = np.arange(len(diffsky_data["redshift_true"])).astype(
        int
    )
    for key in diffsky_data["mah_params"]._fields:
        diffsky_data[key] = getattr(diffsky_data["mah_params"], key)

    diffsky_data["logmp_obs"] = logmh_at_t_obs(
        diffsky_data["mah_params"], diffsky_data["t_obs"], sim_info.lgt0
    )
    diffsky_data["logmp_obs_host"] = diffsky_data["logmp_obs"][
        diffsky_data["top_host_idx"]
    ]

    diffsky_data.pop("mah_params")

    posvel_collector = ("x", "y", "z", "vx", "vy", "vz")

    eigh_pat = "top_host_infall_fof_halo_eigS{0}{1}"
    eigh_collector = []
    for n in (1, 2, 3):
        for s in ("X", "Y", "Z"):
            eigh_collector.append(eigh_pat.format(n, s))
    for key in (
        *posvel_collector,
        "x_host",
        "y_host",
        "z_host",
        "ra",
        "dec",
        *eigh_collector,
    ):
        diffsky_data[key] = np.zeros(len(diffsky_data["redshift_true"])) - 1.0

    n_gals = len(diffsky_data["redshift_true"])
    ZZ = np.zeros(n_gals)

    ran_key, pos_key, vel_key = jran.split(ran_key, 3)
    pos = jran.uniform(pos_key, minval=-1000, maxval=-999.9, shape=(n_gals, 3))
    diffsky_data["x"] = pos[:, 0]
    diffsky_data["y"] = pos[:, 1]
    diffsky_data["z"] = pos[:, 2]
    diffsky_data["x_host"] = pos[:, 0]
    diffsky_data["y_host"] = pos[:, 1]
    diffsky_data["z_host"] = pos[:, 2]

    _res = lc_mock.get_imputed_velocity(ZZ, ZZ, ZZ, vel_key)
    vx_imputed, vy_imputed, vz_imputed, msk_imputed = _res
    diffsky_data["vx"] = vx_imputed
    diffsky_data["vy"] = vy_imputed
    diffsky_data["vz"] = vz_imputed

    diffsky_data["core_tag"] = -np.ones(len(diffsky_data["redshift_true"])).astype(int)
    diffsky_data["has_diffmah_fit"] = ZZ.astype(int) + 1
    diffsky_data["n_points_per_fit"] = ZZ.astype(int) + 10_000
    diffsky_data["loss"] = ZZ + 1e-5
    diffsky_data["central"] = ZZ.astype(int) + 1

    diffsky_data["theta"] = ZZ - 1.0
    diffsky_data["phi"] = ZZ - 1.0
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
