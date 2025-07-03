""" """

import os

import numpy as np

from .. import load_flat_hdf5

BNPAT_LC_MOCK = "lc_cores-{0}.{1}.diffsky_gals.hdf5"


def check_all_columns_are_finite(fn):

    msg = []
    mock = load_flat_hdf5(fn, dataset="data")
    bn = os.path.basename(fn)
    for key, arr in mock.items():
        if not np.all(np.isfinite(arr)):
            s = f"Column {key} in {bn} has either NaN or inf"
            msg.append(s)

    return msg
