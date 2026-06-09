""" """

import argparse
import os
from glob import glob
from time import time

import h5py
import numpy as np
import yaml
from astropy import units as u

from diffsky.data_loaders.mock_utils import get_mock_version_name

GALAXY_ID_COLNAME = "gal_id"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml", help="YAML configuration file")
    args = parser.parse_args()

    cl_args = parser.parse_args()
    config_path = cl_args.config_yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    drn_out = config["drn_out"]
    mock_nickname = config["mock_nickname"]
    mock_version_name = get_mock_version_name(mock_nickname)
    drn = os.path.join(drn_out, mock_version_name)

    fn_list = glob(os.path.join(drn, "lc_cores-*.hdf5"))
    bn_list = [os.path.basename(fn) for fn in fn_list]

    def sort_key(bn):
        bn_parse = bn.split("-")[1].split(".")
        stepnum = int(bn_parse[0])
        lc_patch = int(bn_parse[1])
        synth_halo = int(len(bn_parse[2:]) > 2)
        return (stepnum, lc_patch, synth_halo)

    bn_list_sorted = sorted(bn_list, key=sort_key)
    n_files = len(bn_list_sorted)

    start = time()
    counter = 0
    for bn in bn_list_sorted:
        fn = os.path.join(drn, bn)
        with h5py.File(fn, "r") as hdf:
            n_fn = hdf["data/central"].size

        print(f"Adding {GALAXY_ID_COLNAME} to {bn}")
        with h5py.File(fn, "a") as hdf:
            galid_fn = np.arange(counter, counter + n_fn).astype(int)
            key = "/".join(("data", GALAXY_ID_COLNAME))
            if key in hdf:
                del hdf[key]
            hdf[key] = galid_fn
            hdf[key].attrs["unit"] = str(u.dimensionless_unscaled)
            hdf[key].attrs["description"] = "Unique integer ID for each galaxy"

        counter += n_fn

    end = time()

    max_id = galid_fn[-1]
    runtime_sec = end - start
    runtime = runtime_sec / 60.0
    print(f"Max value of {GALAXY_ID_COLNAME} = {max_id:_}")
    print(f"Total runtime to add galaxy ids to {n_files} = {runtime:.1f} minutes\n")
