""""""

import os
from copy import deepcopy

import h5py
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS

from . import hacc_core_utils as hcu
from . import load_hacc_cores as lhc

DEG_PER_RAD = 180 / np.pi
SQDEG_PER_STER = DEG_PER_RAD**2
SQDEG_OF_SPHERE = SQDEG_PER_STER * 4 * np.pi

TOP_HOST_MAH_KEYS = ["top_host_" + key for key in DEFAULT_MAH_PARAMS._fields]
SECONDARY_HOST_MAH_KEYS = ["sec_host_" + key for key in DEFAULT_MAH_PARAMS._fields]

LC_PATCH_OUT_KEYS = (
    *DEFAULT_MAH_PARAMS._fields,
    *TOP_HOST_MAH_KEYS,
    *SECONDARY_HOST_MAH_KEYS,
    "loss",
    "n_points_per_fit",
    "indx_t_ult_inf",
    "indx_t_pen_inf",
    "n_cf_match",
)
LC_PATCH_OUT_INT_KEYS = (
    "n_points_per_fit",
    "indx_t_ult_inf",
    "indx_t_pen_inf",
    "n_cf_match",
)
LC_SUBVOL_DRNPAT = "subvols_*"
LC_PATCH_DIFFSKY_BNPAT = "lc_cores-{0}.{1}.diffsky_data.hdf5"
LC_PATCH_BNPAT = "lc_cores-{0}.{1}.hdf5"


def read_lc_ra_dec_patch_decomposition(fn):
    collector = []
    with open(fn, "r") as f:
        next(f)
        next(f)
        for raw_line in f:
            line = raw_line.strip().split()
            patch_idx = int(line[0])
            theta_low = float(line[1][1:-1])
            theta_high = float(line[2][:-1])
            phi_low = float(line[3][1:-1])
            phi_high = float(line[4][:-1])
            data = patch_idx, theta_low, theta_high, phi_low, phi_high
            collector.append(data)
    patch_decomposition = np.array(collector)

    sky_frac = []
    solid_angles = []
    for ipatch in range(patch_decomposition.shape[0]):
        dec_min, dec_max, ra_min, ra_max = patch_decomposition[ipatch, 1:]
        solid_angle, fraction = calculate_solid_angle(ra_min, ra_max, dec_min, dec_max)
        sky_frac.append(fraction)
        solid_angles.append(solid_angle)
    sky_frac = np.array(sky_frac)
    solid_angles = np.array(solid_angles)

    return patch_decomposition, sky_frac, solid_angles


def calculate_solid_angle(ra_min, ra_max, dec_min, dec_max, epsilon=0.001):
    """
    Calculate the solid angle of a rectangular patch of sky.

    Parameters:
    -----------
    ra_min, ra_max : floats
        Right ascension range in radians (0 to 2π)

    dec_min, dec_max : floats
        Declination range in radians (0 to π, where 0 is north pole,
        π/2 is equator, and π is south pole)

    Returns:
    --------
    solid_angle : float
        Solid angle in square degrees

    fsky : float
        Fraction of the full sky covered

    """
    # Input validation
    ra_lo, ra_hi = -epsilon, 2 * np.pi + epsilon
    dec_lo, dec_hi = -epsilon, np.pi + epsilon

    if not (ra_lo <= ra_min <= ra_hi and ra_lo <= ra_max <= ra_hi):
        raise ValueError(
            f"RA must be between 0 and 2π radians. Got ra_min={ra_min}, ra_max={ra_max}"
        )
    if not (dec_lo <= dec_min <= dec_hi and dec_lo <= dec_max <= dec_hi):
        raise ValueError(
            f"DEC must be between 0 and π radians. Got dec_min={dec_min}, dec_max={dec_max}"
        )
    if ra_min > ra_max:
        raise ValueError(
            f"ra_min must be less than ra_max. Got ra_min={ra_min}, ra_max={ra_max}"
        )
    if dec_min > dec_max:
        raise ValueError(
            f"dec_min must be less than dec_max. Got dec_min={dec_min}, dec_max={dec_max}"
        )

    ra_min = np.clip(ra_min, 0, 2 * np.pi)
    ra_max = np.clip(ra_max, 0, 2 * np.pi)
    dec_min = np.clip(dec_min, 0, np.pi)
    dec_max = np.clip(dec_max, 0, np.pi)

    delta_ra = ra_max - ra_min
    solid_angle = delta_ra * (np.cos(dec_min) - np.cos(dec_max))  # in steradians
    solid_angle = solid_angle * SQDEG_PER_STER

    # Full sky is 4π steradians
    full_sky = 4 * np.pi
    fsky = solid_angle / full_sky

    return solid_angle, fsky


def get_infall_times_lc_shell(forest_matrices, timestep_idx):
    args = (
        forest_matrices["host_row"],
        forest_matrices["central"],
        forest_matrices["top_host_row"],
        forest_matrices["secondary_top_host_row"],
        timestep_idx,
    )
    indx_t_ult_inf, indx_t_pen_inf = lhc.get_infall_time_indices(*args)
    return indx_t_ult_inf, indx_t_pen_inf


def get_diffsky_quantities_for_lc_patch(
    lc_patch_data,
    lc_patch_data_out,
    cf_matrices,
    cf_diffmah_data,
    cf_file_idx,
    cf_indx_t_ult_inf,
    cf_indx_t_pen_inf,
    timestep_idx,
):
    cf_first_row = cf_matrices["absolute_row_idx"][0]
    cf_last_row = cf_matrices["absolute_row_idx"][-1]
    msk_olap = lc_patch_data["file_idx"] == cf_file_idx
    msk_olap &= lc_patch_data["row_idx"] >= cf_first_row
    msk_olap &= lc_patch_data["row_idx"] <= cf_last_row

    n_olap = msk_olap.sum()
    if n_olap > 0:
        olap_chunk_idx = lc_patch_data["row_idx"][msk_olap] - cf_first_row
        olap_chunk_ult_host_idx = cf_matrices["top_host_row"][:, timestep_idx][
            olap_chunk_idx
        ]
        olap_chunk_pen_host_idx = cf_matrices["secondary_top_host_row"][
            :, timestep_idx
        ][olap_chunk_idx]

        olap_indx_t_ult_inf = cf_indx_t_ult_inf[olap_chunk_idx]
        lc_patch_data_out["indx_t_ult_inf"][msk_olap] = olap_indx_t_ult_inf

        olap_indx_t_pen_inf = cf_indx_t_pen_inf[olap_chunk_idx]
        lc_patch_data_out["indx_t_pen_inf"][msk_olap] = olap_indx_t_pen_inf

        lc_patch_data_out["n_cf_match"][msk_olap] += 1

        # Matchup mah_params
        _keys = (*DEFAULT_MAH_PARAMS._fields, "loss", "n_points_per_fit")
        _olap_data = [cf_diffmah_data[key][olap_chunk_idx] for key in _keys]
        for key, x in zip(_keys, _olap_data):
            lc_patch_data_out[key][msk_olap] = x

        # Matchup uber host_mah_params
        mah_keys = DEFAULT_MAH_PARAMS._fields
        host_mah_keys = TOP_HOST_MAH_KEYS
        _olap_ult_host_data = [
            cf_diffmah_data[key][olap_chunk_ult_host_idx] for key in mah_keys
        ]
        for host_key, x in zip(host_mah_keys, _olap_ult_host_data):
            lc_patch_data_out[host_key][msk_olap] = x

        # Matchup secondary host_mah_params
        mah_keys = DEFAULT_MAH_PARAMS._fields
        second_host_mah_keys = SECONDARY_HOST_MAH_KEYS
        _olap_pen_host_data = [
            cf_diffmah_data[key][olap_chunk_pen_host_idx] for key in mah_keys
        ]
        for host_key, x in zip(second_host_mah_keys, _olap_pen_host_data):
            lc_patch_data_out[host_key][msk_olap] = x

    return lc_patch_data_out


def _get_lc_patch_data_out_bname(bn_patch):
    bn_out = bn_patch.replace(".hdf5", ".diffsky_data.hdf5")
    return bn_out


def load_lc_patch_data_out(drn, bn_patch):
    bn_out = _get_lc_patch_data_out_bname(bn_patch)
    fn_out = os.path.join(drn, bn_out)
    lc_patch_data_out = hcu.load_flat_hdf5(fn_out)
    return lc_patch_data_out


def initialize_lc_patch_data_out(n_patch):
    lc_patch_data_out = dict()
    for key in LC_PATCH_OUT_KEYS:
        if key in LC_PATCH_OUT_INT_KEYS:
            lc_patch_data_out[key] = np.zeros(n_patch).astype(int)
        else:
            lc_patch_data_out[key] = np.zeros(n_patch).astype(float)

    return lc_patch_data_out


def overwrite_lc_patch_data_out(lc_patch_data_out, drn_out, bn_patch):
    bn_out = _get_lc_patch_data_out_bname(bn_patch)
    fn_out = os.path.join(drn_out, bn_out)

    with h5py.File(fn_out, "w") as hdf_out:
        for key in lc_patch_data_out.keys():
            hdf_out[key] = lc_patch_data_out[key]


def get_stepnum_and_skypatch_from_lc_bname(lc_bname):
    """Assumes basename format such as lc_cores-300.0.hdf5"""
    seq = lc_bname.split("-")[1].split(".")
    stepnum = int(seq[0])
    patchnum = int(seq[1])
    return stepnum, patchnum


def get_lc_patches_in_zrange(sim_name, lc_xdict, z_min, z_max, patch_list=None):
    _res = hcu.get_timestep_range_from_z_range(sim_name, z_min, z_max)
    timestep_min, timestep_max = _res[2:]

    lc_patches = []
    for patch_info in lc_xdict.keys():
        stepnum, patchnum = patch_info

        if (stepnum >= timestep_min) & (stepnum <= timestep_max):

            if patch_list is None:
                lc_patches.append(patch_info)
            else:
                if patchnum in patch_list:
                    lc_patches.append(patch_info)

    return lc_patches


def check_lc_cores_diffsky_data(fn):
    report = dict()
    data = hcu.load_flat_hdf5(fn)
    n_cf_match_list = np.unique(data["n_cf_match"])
    report["n_cf_match_list"] = n_cf_match_list
    return report


def write_lc_cores_diffsky_data_report_to_disk(report, fnout):
    lc_cf_perfect_match = set(report["n_cf_match_list"]) == {1}
    if not lc_cf_perfect_match:
        fnout = fnout.replace(".hdf5", ".report.txt")
        with open(fnout, "w") as fout:
            fout.write("n_cf_match is not 1 for all objects\n")

    all_good = deepcopy(lc_cf_perfect_match)

    return all_good
