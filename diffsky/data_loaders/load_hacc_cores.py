"""General purpose data-loading utilities for HACC cores"""

import os

BNPAT = "m000p.coreforest.{}.hdf5"


def _get_all_avail_basenames(drn, pat, subvolumes):
    fname_list = [os.join(drn, pat.format(i)) for i in subvolumes]
    for fn in fname_list:
        assert os.path.isfile(fn)
    return fname_list
