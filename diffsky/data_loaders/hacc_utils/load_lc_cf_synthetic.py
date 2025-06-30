""" """

import os

import numpy as np
from diffmah import logmh_at_t_obs

from ...experimental import mc_lightcone_halos as mclh
from . import lightcone_utils as hlu


def load_lc_diffsky_patch_synthetic_data(
    fn_lc_cores, sim_name, ran_key, lgmp_min, lgmp_max
):
    drn_lc_cores = os.path.dirname(fn_lc_cores)
    bname_lc_cores = os.path.basename(fn_lc_cores)
    lc_patch = int(bname_lc_cores.split("-")[1].split(".")[:-1][1])

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
    diffsky_data["z_true"] = diffsky_data["z_obs"]
    del diffsky_data["z_obs"]

    diffsky_data["top_host_idx"] = np.arange(len(diffsky_data["z_true"])).astype(int)
    for key in diffsky_data["mah_params"]._fields:
        diffsky_data[key] = getattr(diffsky_data["mah_params"], key)

    diffsky_data["logmp_obs"] = logmh_at_t_obs(
        diffsky_data["mah_params"], diffsky_data["t_obs"], 1.14
    )
    diffsky_data["logmp_obs_host"] = diffsky_data["logmp_obs"][
        diffsky_data["top_host_idx"]
    ]

    diffsky_data.pop("mah_params")

    for key in ("x", "y", "z", "x_host", "y_host", "z_host", "ra", "dec"):
        diffsky_data[key] = np.zeros(len(diffsky_data["z_true"])) - 1.0

    diffsky_data["core_tag"] = -np.ones(len(diffsky_data["z_true"])).astype(int)
    diffsky_data["has_diffmah_fit"] = 0
    diffsky_data["central"] = 1

    lc_data = diffsky_data
    return lc_data, diffsky_data
