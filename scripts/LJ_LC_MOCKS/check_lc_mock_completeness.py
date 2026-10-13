""""""

import argparse
import os
from glob import glob

import numpy as np
import yaml

from diffsky.data_loaders.hacc_utils import lc_mock
from diffsky.data_loaders.hacc_utils import lightcone_utils as hlu

BN_PAT_REAL_HALOS = "lc_cores-{0}.{1}.diffsky_gals.hdf5"
BN_PAT_REAL_OR_SYNTHETIC = "lc_cores-{0}.{1}.*.hdf5"


def get_stepnum_and_patch(bname):
    _stepnum, _patch = bname.split("-")[1].split(".")[:2]
    stepnum = int(_stepnum)
    patch = int(_patch)
    return stepnum, patch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_yaml",
        help="Name of the config file."
        "Automatically determines drn_mock unless overrided by drn_out or infer_drn_mock",
    )

    parser.add_argument(
        "-drn_out", help="Absolute path to directory of mock.", default=""
    )
    parser.add_argument(
        "-infer_drn_mock",
        help="Location of mock determined by appending mock_version_name to config['drn_out']",
        action="store_true",
    )
    parser.add_argument(
        "-z_min",
        help="Minimum expected redshift. Overrides config_yaml",
        type=float,
        default=-1,
    )
    parser.add_argument(
        "-z_max",
        help="Maximum expected redshift. Overrides config_yaml",
        type=float,
        default=-1,
    )

    cl_args = parser.parse_args()

    config_yaml = cl_args.config_yaml
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    # Determine location of mock
    if cl_args.drn_out != "":
        drn_out = cl_args.drn_out
    elif cl_args.infer_drn_mock:
        _fn_list = glob(os.path.join(config["drn_out"], config["mock_nickname"] + "_*"))
        mock_version_name = os.path.basename(_fn_list[0])
        drn_out = os.path.join(config["drn_out"], mock_version_name)
    else:
        drn_out = config["drn_out"]

    if cl_args.z_min == -1:
        z_min = config["z_min"]
    else:
        z_min = cl_args.z_min
    if cl_args.z_max == -1:
        z_max = config["z_max"]
    else:
        z_max = cl_args.z_max

    # Check completeness of simulated halos
    fn_pat = os.path.join(drn_out, BN_PAT_REAL_HALOS.format("*", "*"))
    fn_list = glob(fn_pat)
    assert len(fn_list) > 0, f"Zero files matching {fn_pat}"
    bn_list = [os.path.basename(fn) for fn in fn_list]

    step_collector = np.unique([get_stepnum_and_patch(bn)[0] for bn in bn_list])
    patch_collector = np.unique([get_stepnum_and_patch(bn)[1] for bn in bn_list])
    n_patches = len(patch_collector)

    complete_stepnums = hlu.get_timesteps_in_zrange("LastJourney", z_min, z_max)

    print(
        f"Running checks on each of the {n_patches} sky patches has a complete set of redshifts"
    )
    msg_patch = "lc_patch={0} is missing some timesteps\n"
    msg_collector = []
    for patch in patch_collector:
        fn_pat_patch = os.path.join(drn_out, BN_PAT_REAL_HALOS.format("*", patch))
        fn_list_patch = glob(fn_pat_patch)
        bn_list_patch = [os.path.basename(fn) for fn in fn_list_patch]
        steps_patch = np.unique([get_stepnum_and_patch(bn)[0] for bn in bn_list_patch])
        try:
            assert len(steps_patch) == len(complete_stepnums)
            assert np.allclose(steps_patch, complete_stepnums)
        except AssertionError:
            msg_collector.append(msg_patch.format(patch))
    msg_success = f"\nDetected complete set of mocks with real halos spanning {z_min:.2f}<z<{z_max:.2f}\n"
    if len(msg_collector) > 0:
        for msg in msg_collector:
            print(msg)
    else:
        print(msg_success)

    # Check each lightcone file has a matching synthetic_halos file
    unmatched_bn_list = []
    for fn in fn_list:
        bn = os.path.basename(fn)
        bn_synthetic = bn.replace(".hdf5", ".synthetic_halos.hdf5")
        fn_synthetic = os.path.join(drn_out, bn_synthetic)
        if not os.path.isfile(fn_synthetic):
            unmatched_bn_list.append(bn)

    if len(unmatched_bn_list) > 0:
        msg = "\nMissing synthetic_halos for some lightcone files\n"
        msg += f"Example: {unmatched_bn_list[0]} has no synthetic_halos counterpart"
        print(msg)
    else:
        print("\nEach lc_cores file has a synthetic_halos matching counterpart")

    # Check each mock file finished writing
    fnpat = os.path.join(drn_out, BN_PAT_REAL_OR_SYNTHETIC.format("*", "*"))
    fn_list_all = glob(fnpat)
    n_files = len(fn_list_all)
    assert n_files > 0, f"{fnpat} is an incorrect path to mock dataset"

    incomplete_files = []
    for fn in fn_list_all:
        fn_exists, fn_is_complete = lc_mock.check_mock_data_file_is_complete(fn)
        if not fn_is_complete:
            incomplete_files.append(fn)

    if len(incomplete_files) > 0:
        print(
            "\n...The following files are missing the `write_complete` metadata attribute\n"
        )
        for fn in incomplete_files:
            print(f"{fn}\n")
    else:
        print(
            f"\nEach of the {n_files} detected mock files has the `write_complete` metadata attribute"
        )
