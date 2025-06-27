""" """

import os

import numpy as np

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


def check_core_tag_uniqueness(fn_lc_cores, sim_name, tol=0.0002, lc_cores=None):
    msg = []
    if lc_cores is None:
        lc_cores = load_flat_hdf5(fn_lc_cores)
        u_core_tags, counts = np.unique(lc_cores["core_tag"], return_counts=True)
        if u_core_tags.size < lc_cores["core_tag"].size:
            example_repeated_core_tag = u_core_tags[counts > 1][0]
            s = f"repeated core tag = {example_repeated_core_tag}"
            msg.append(s)
            n_distinct_repeats = np.sum(counts > 1)
            max_repetitions = counts.max()
            s = f"Number of distinct repeats = {n_distinct_repeats}"
            msg.append(s)
            s = f"Max num repetitions = {max_repetitions}"
            msg.append(s)
    return msg
