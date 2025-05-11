"""Utility functions for reading and writing data"""

import h5py


def load_flat_hdf5(fn, istart=0, iend=None, keys=None):
    """"""

    data = dict()
    with h5py.File(fn, "r") as hdf:

        if keys is None:
            keys = list(hdf.keys())

        for key in keys:
            if iend is None:
                data[key] = hdf[key][istart:]
            else:
                data[key] = hdf[key][istart:iend]

    return data
