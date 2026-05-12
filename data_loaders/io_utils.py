"""Utility functions for reading and writing data"""

from collections import namedtuple

import h5py
import numpy as np


def load_flat_hdf5(fn, istart=0, iend=None, keys=None, dataset=None):
    """Load flat hdf5 file

    Parameters
    ----------
    fn : string

    istart : int, optional
        First row of data to read

    iend : int, optional
        Last row of data to read

    keys : list, optional
        List of strings

    dataset : string, optional
        Dataset within hdf5 file

    Returns
    -------
    data : dict

    """

    data = dict()
    with h5py.File(fn, "r") as hdf:

        if dataset is None:
            if keys is None:
                keys_in = list(hdf.keys())
            else:
                keys_in = keys
        else:
            if keys is None:
                keys_in = [dataset + "/" + key for key in list(hdf[dataset].keys())]
            else:
                keys_in = [dataset + "/" + key for key in keys]

        for key_in in keys_in:
            if dataset is None:
                key_out = key_in
            else:
                key_out = key_in.split("/")[1]

            if iend is None:
                data[key_out] = hdf[key_in][istart:]
            else:
                data[key_out] = hdf[key_in][istart:iend]

    return data


def write_namedtuple_to_hdf5(named_params, fn_out):
    """Write an hdf5 file to a namedtuple"""
    with h5py.File(fn_out, "w") as hdf:
        for pname, pval in zip(named_params._fields, named_params):
            hdf[pname] = pval
        hdf.attrs["_fields"] = np.array(named_params._fields, dtype="S")


def load_namedtuple_from_hdf5(fn):
    """Load an hdf5 file storing a namedtuple"""
    with h5py.File(fn, "r") as hdf:
        pnames = [
            f.decode() if isinstance(f, bytes) else f for f in hdf.attrs["_fields"]
        ]
        values = [hdf[pname][()] for pname in pnames]
    Params = namedtuple("Params", pnames)
    params = Params(*values)
    return params
