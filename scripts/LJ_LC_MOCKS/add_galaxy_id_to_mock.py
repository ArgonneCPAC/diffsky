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
    parser.add_argument(
        "-drn_mock", help="Directory of mock overrides config_yaml", default=""
    )
    parser.add_argument(
        "-infer_mockname",
        help="Infer mock_version_name from directory",
        action="store_true",
    )

    args = parser.parse_args()

    cl_args = parser.parse_args()
    config_path = cl_args.config_yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mock_nickname = config["mock_nickname"]

    mock_version_name_in = config.get("mock_version_name", "")
    if cl_args.infer_mockname:
        fnpat = os.path.join(config["drn_out"], mock_nickname + "_*")
        fn_list = glob(fnpat)
        try:
            drn_mock = fn_list[0]
        except IndexError:
            raise ValueError(f"No files detected in {fnpat}")
        mock_version_name = os.path.basename(drn_mock)
    else:
        if mock_version_name_in == "":
            mock_version_name = get_mock_version_name(mock_nickname)
        else:
            mock_version_name = mock_version_name_in

    if cl_args.drn_mock == "":
        drn_mock = os.path.join(config["drn_out"], mock_version_name)
    else:
        drn_mock = cl_args.drn_mock

    fn_list = glob(os.path.join(drn_mock, "lc_cores-*.hdf5"))
    bn_list = [os.path.basename(fn) for fn in fn_list]
    assert len(bn_list) > 0, f"No mock files detected in {drn_mock}"

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
        fn = os.path.join(drn_mock, bn)
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
