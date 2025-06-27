""" """

import os

import numpy as np

from .. import lightcone_utils as lcu
from .. import load_flat_hdf5

try:
    from haccytrees import Simulation as HACCSim

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False

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
