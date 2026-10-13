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
    parser.add_argument("drn_out", help="Location of mock.")

    parser.add_argument(
        "-config_yaml",
        help="Name of the config file. Overrides automatic detection",
        default="",
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
    drn_out = cl_args.drn_out

    if cl_args.config_yaml != "":
        config_yaml = cl_args.config_yaml
    else:
        # Automatically detect config_yaml
        config_yaml_list = glob(os.path.join(drn_out, "*.yaml"))
        if len(config_yaml_list) == 1:
            config_yaml = config_yaml_list[0]
        elif len(config_yaml_list) == 0:
            raise ValueError(
                f"Missing config_yaml in {drn_out} - must specify with command line argument"
            )
        elif len(config_yaml_list) > 1:
            raise ValueError(
                f"Multiple detectiosn of config_yaml in {drn_out} - must disambiguate with command line argument"
            )

    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

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
    msg_patch = f"lc_patch={0} is missing some timesteps"
    for patch in patch_collector:
        fn_pat_patch = os.path.join(drn_out, BN_PAT_REAL_HALOS.format("*", patch))
        fn_list_patch = glob(fn_pat_patch)
        bn_list_patch = [os.path.basename(fn) for fn in fn_list_patch]
        steps_patch = np.unique([get_stepnum_and_patch(bn)[0] for bn in bn_list_patch])
        assert len(steps_patch) == len(complete_stepnums), msg_patch.format(patch)
        assert np.allclose(steps_patch, complete_stepnums), msg_patch.format(patch)
    print(f"\nDetected complete set of lc_cores spanning {z_min:.2f}<z<{z_max:.2f}\n")

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
    assert len(fn_list_all) > 0, f"{fnpat} is an incorrect path to mock dataset"
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
