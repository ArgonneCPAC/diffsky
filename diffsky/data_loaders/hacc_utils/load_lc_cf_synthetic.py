""" """

import os

import numpy as np
from diffmah import logmh_at_t_obs

from ...experimental import mc_lightcone_halos as mclh
from . import lightcone_utils as hlu
from . import load_lc_cf as llcf


def load_lc_diffsky_patch_data(fn_lc_cores, sim_name, ran_key, lgmp_min, lgmp_max):

    sim_info = llcf.get_diffsky_info_from_hacc_sim(sim_name)

    drn_lc_cores = os.path.dirname(fn_lc_cores)
    bname_lc_cores = os.path.basename(fn_lc_cores)
    snapnum, lc_patch = hlu.get_stepnum_and_skypatch_from_lc_bname(bname_lc_cores)

    _res = hlu.read_lc_ra_dec_patch_decomposition(
        os.path.join(drn_lc_cores, "lc_cores-decomposition.txt")
    )
    patch_decomposition, sky_frac, solid_angles = _res
    sky_area_degsq = solid_angles[lc_patch]

    a_min, a_max = hlu.get_a_range_of_lc_cores_file(bname_lc_cores, sim_name)

    z_min = 1 / a_max - 1
    z_max = 1 / a_min - 1
    args = (
        ran_key,
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

    diffsky_data["top_host_idx"] = np.arange(len(diffsky_data["redshift_true"])).astype(int)
    for key in diffsky_data["mah_params"]._fields:
        diffsky_data[key] = getattr(diffsky_data["mah_params"], key)

    diffsky_data["logmp_obs"] = logmh_at_t_obs(
        diffsky_data["mah_params"], diffsky_data["t_obs"], sim_info.lgt0
    )
    diffsky_data["logmp_obs_host"] = diffsky_data["logmp_obs"][
        diffsky_data["top_host_idx"]
    ]

    diffsky_data.pop("mah_params")

    for key in ("x", "y", "z", "x_host", "y_host", "z_host", "ra", "dec"):
        diffsky_data[key] = np.zeros(len(diffsky_data["redshift_true"])) - 1.0

    ZZ = np.zeros(len(diffsky_data["redshift_true"]))
    diffsky_data["core_tag"] = -np.ones(len(diffsky_data["redshift_true"])).astype(int)
    diffsky_data["has_diffmah_fit"] = ZZ.astype(int) + 1
    diffsky_data["n_points_per_fit"] = ZZ.astype(int) + 10_000
    diffsky_data["loss"] = ZZ + 1e-5
    diffsky_data["central"] = ZZ.astype(int) + 1

    diffsky_data["theta"] = ZZ - 1.0
    diffsky_data["phi"] = ZZ - 1.0
    diffsky_data["snapnum"] = ZZ.astype(int) + snapnum

    lc_data = diffsky_data
    return lc_data, diffsky_data
