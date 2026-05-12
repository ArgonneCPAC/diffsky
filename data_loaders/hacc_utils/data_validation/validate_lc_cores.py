""" """

import os

import numpy as np

from ....utils import crossmatch
from .. import lightcone_utils as lcu
from .. import load_flat_hdf5

BNPAT_LC_CORES = "lc_cores-{0}.{1}.hdf5"


def check_zrange(fn_lc_cores, sim_name, tol=0.0002, lc_cores=None):
    """Redshift range of the data should be bounded by the expected range"""

    bn_lc_cores = os.path.basename(fn_lc_cores)
    a_min_expected, a_max_expected = lcu.get_a_range_of_lc_cores_file(
        bn_lc_cores, sim_name
    )

    if lc_cores is None:
        lc_cores = load_flat_hdf5(fn_lc_cores)

    a_min_data = lc_cores["scale_factor"].min()
    a_max_data = lc_cores["scale_factor"].max()

    msg = []
    if a_min_data < a_min_expected - tol:
        msg.append(f"a_min_data={a_min_data} < a_min_expected={a_min_expected}\n")
    if a_max_data > a_max_expected + tol:
        msg.append(f"a_max_data={a_max_data} > a_max_expected={a_max_expected}\n")

    return msg


def check_core_tag_uniqueness(fn_lc_cores, lc_cores=None):
    if lc_cores is None:
        lc_cores = load_flat_hdf5(fn_lc_cores)

    msg = []
    u_core_tags, counts = np.unique(lc_cores["core_tag"], return_counts=True)
    if u_core_tags.size < lc_cores["core_tag"].size:
        example_repeated_core_tag = u_core_tags[counts > 1][0]
        s = f"repeated core_tag = {example_repeated_core_tag}"
        msg.append(s)
        n_distinct_repeats = np.sum(counts > 1)
        max_repetitions = counts.max()
        s = f"Number of distinct repeats = {n_distinct_repeats}"
        msg.append(s)
        s = f"Max num repetitions = {max_repetitions}"
        msg.append(s)
    return msg


def check_top_host_tag_has_match(fn_lc_cores, lc_cores=None):

    if lc_cores is None:
        lc_cores = load_flat_hdf5(fn_lc_cores)

    u_core_tag, u_indx, counts = np.unique(
        lc_cores["core_tag"], return_counts=True, return_index=True
    )

    msg = []
    if u_core_tag.size < lc_cores["core_tag"].size:
        s = "Could not run test of top_host_tag due to repeated values of core_tag"
        msg.append(s)
    else:
        idxA, idxB = crossmatch(lc_cores["top_host_tag"], lc_cores["core_tag"])
        indarr = np.arange(len(u_core_tag)).astype(int)
        indarr[idxA] = indarr[idxB]

        if not np.allclose(indarr, lc_cores["top_host_idx"]):
            msk = indarr != lc_cores["top_host_idx"]
            example_core_tag = lc_cores["core_tag"][msk][0]
            example_top_host_tag = lc_cores["top_host_tag"][msk][0]
            s = f"core_tag = {example_core_tag} has top_host_tag={example_top_host_tag} "
            s += "which disagrees with diffsky.utils.crossmatch"
            msg.append(s)

            n_mismatches = msk.sum()
            s = f"Number of discrepancies between top_host_tag and diffsky = {n_mismatches}"
            msg.append(s)

    return msg


def check_top_host_idx_tag_agreement(fn_lc_cores, lc_cores=None):
    """top_host_tag should always be consistent with top_host_idx"""
    if lc_cores is None:
        lc_cores = load_flat_hdf5(fn_lc_cores)

    msg = []
    top_host_tag2 = lc_cores["core_tag"][lc_cores["top_host_idx"]]
    if not np.allclose(lc_cores["top_host_tag"], top_host_tag2):
        msk = top_host_tag2 != lc_cores["top_host_idx"]
        example_core_tag = lc_cores["core_tag"][msk][0]
        example_top_host_tag = lc_cores["top_host_tag"][msk][0]
        s = f"core_tag = {example_core_tag} has top_host_tag={example_top_host_tag} "
        s += "which disagrees with the value inferred from the `top_host_idx` column"
        msg.append(s)

    return msg


def check_host_pos_is_near_galaxy_pos(fn_lc_cores, lc_cores=None):
    """host position should be reasonably close to galaxy position"""
    if lc_cores is None:
        lc_cores = load_flat_hdf5(fn_lc_cores)

    bn = os.path.basename(fn_lc_cores)

    host_x = lc_cores["x"][lc_cores["top_host_idx"]]
    host_y = lc_cores["y"][lc_cores["top_host_idx"]]
    host_z = lc_cores["z"][lc_cores["top_host_idx"]]
    dx = lc_cores["x"] - host_x
    dy = lc_cores["y"] - host_y
    dz = lc_cores["z"] - host_z
    host_dist = np.sqrt(dx**2 + dy**2 + dz**2)

    msg = []
    n_very_far = np.sum(host_dist > 5)
    if n_very_far > 10:
        s = f"{n_very_far} galaxies in {bn} with "
        s += "unexpectedly large xyz distance from top_host_idx"
        msg.append(s)

    msk_cen = lc_cores["central"]
    mean_sat_dist = np.abs(np.mean(host_dist[~msk_cen]))
    std_sat_dist = np.std(host_dist[~msk_cen])
    if mean_sat_dist > 1:
        s = f"<dist_sat>={mean_sat_dist:.2f} Mpc/h is unexpectedly large"
        msg.append(s)
    if std_sat_dist > 0.5:
        s = f"std(dist_sat)={std_sat_dist:.2f} Mpc/h is unexpectedly large"
        msg.append(s)

    return msg
