""""""

import argparse
import os
import pickle
from glob import glob

import numpy as np

from diffsky.data_loaders.hacc_utils import lightcone_utils as hlu

BNPAT = "lc_cores-{0}.{1}.diffsky_data.hdf5"

SIM_NAME = "LastJourney"
DRN_LC_CF = "/lcrc/project/cosmo_ai/ahearin/LastJourney/lc-cf-diffsky"


def get_stepnum_and_patch(bname):
    _stepnum, _patch = bname.split("-")[1].split(".")[:2]
    stepnum = int(_stepnum)
    patch = int(_patch)
    return stepnum, patch


def compute_missing_patches(step_match_info):
    print("...computing missing_patches")
    uniq_steps = np.sort(list(step_match_info.keys()))

    missing_patches = dict()
    for stepnum in uniq_steps:
        avail_patches = step_match_info[stepnum]
        max_patch = np.max(avail_patches)
        complete_patches = np.arange(max_patch + 1).astype(int)
        _s = list(set(complete_patches) - set(avail_patches))
        if len(_s) > 0:
            missing_patches[int(stepnum)] = [int(s) for s in _s]

    return missing_patches


def get_patches_with_missing_stepnums(patch_match_info, z_min, z_max):
    uniq_patches = np.sort(list(patch_match_info.keys()))
    complete_stepnums = hlu.get_timesteps_in_zrange(SIM_NAME, z_min, z_max)

    print("...computing patches_with_missing_stepnums")
    patches_with_missing_stepnums = dict()
    for patch in uniq_patches:
        avail_stepnums = patch_match_info[patch]
        _s = list(set(complete_stepnums) - set(avail_stepnums))
        if len(_s) > 0:
            patches_with_missing_stepnums[int(patch)] = [int(s) for s in _s]

    return patches_with_missing_stepnums


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("drn", help="Name of the directory storing cross-x files")
    parser.add_argument("z_min", help="Minimum expected redshift", type=float)
    parser.add_argument("z_max", help="Maximum expected reshift", type=float)
    args = parser.parse_args()
    drn = args.drn
    z_min = args.z_min
    z_max = args.z_max

    fn_pat = os.path.join(drn, BNPAT.format("*", "*"))
    bn_list = [os.path.basename(fn) for fn in glob(fn_pat)]
    patch_list = []
    step_list = []

    print("...computing step & patch of all available files")
    for bn in bn_list:
        stepnum, patch = get_stepnum_and_patch(bn)
        patch_list.append(patch)
        step_list.append(stepnum)

    uniq_patches = np.unique(patch_list)
    uniq_steps = np.unique(step_list)[::-1]

    patch_match_info = dict()
    step_match_info = dict()

    for patch in uniq_patches:
        fn_pat = os.path.join(drn, BNPAT.format("*", patch))
        bn_list_patch = [os.path.basename(fn) for fn in glob(fn_pat)]
        stepnum_list_patch = [get_stepnum_and_patch(bn)[0] for bn in bn_list_patch]
        stepnum_list_patch = [int(x) for x in sorted(stepnum_list_patch)[::-1]]
        patch_match_info[int(patch)] = stepnum_list_patch

    print("...computing step_match_info")
    for stepnum in uniq_steps:
        fn_pat = os.path.join(drn, BNPAT.format(stepnum, "*"))
        bn_list_stepnum = [os.path.basename(fn) for fn in glob(fn_pat)]
        patch_list_stepnum = [get_stepnum_and_patch(bn)[1] for bn in bn_list_stepnum]
        patch_list_stepnum = [int(x) for x in sorted(patch_list_stepnum)]
        step_match_info[int(stepnum)] = patch_list_stepnum

    with open("patch_match_info.pickle", "wb") as handle:
        pickle.dump(patch_match_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("step_match_info.pickle", "wb") as handle:
        pickle.dump(step_match_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    missing_patches = compute_missing_patches(step_match_info)
    patches_with_missing_stepnums = get_patches_with_missing_stepnums(
        patch_match_info, z_min, z_max
    )

    with open("missing_patches.pickle", "wb") as handle:
        pickle.dump(missing_patches, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("patches_with_missing_stepnums.pickle", "wb") as handle:
        pickle.dump(
            patches_with_missing_stepnums, handle, protocol=pickle.HIGHEST_PROTOCOL
        )
