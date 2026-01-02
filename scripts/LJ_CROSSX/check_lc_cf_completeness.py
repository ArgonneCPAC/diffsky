""""""

import argparse
import os
import pickle
from glob import glob

import numpy as np

from diffsky.data_loaders import load_flat_hdf5
from diffsky.data_loaders.hacc_utils import lightcone_utils as hlu

BNPAT = "lc_cores-{0}.{1}.diffsky_data.hdf5"

SIM_NAME = "LastJourney"
DRN_LC_CF = "/lcrc/project/cosmo_ai/ahearin/LastJourney/lc-cf-diffsky"
DRN_LC_CORES = (
    "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/core-lc-6/output"
)
FN_LC_CORES = os.path.join(DRN_LC_CORES, "lc_cores-decomposition.txt")


def get_stepnum_and_patch(bname):
    _stepnum, _patch = bname.split("-")[1].split(".")[:2]
    stepnum = int(_stepnum)
    patch = int(_patch)
    return stepnum, patch


def get_stepnums_with_missing_patches(
    step_match_info, complete_stepnums, lc_patches_expected
):
    stepnums_with_missing_patches = dict()
    for stepnum in complete_stepnums:
        try:
            avail_patches = step_match_info[stepnum]
        except KeyError:
            avail_patches = []

        missing_patches = list(set(lc_patches_expected) - set(avail_patches))
        if len(missing_patches) > 0:
            missing_patches = sorted([int(s) for s in missing_patches])
            stepnums_with_missing_patches[int(stepnum)] = missing_patches

    return stepnums_with_missing_patches


def get_patches_with_missing_stepnums(
    patch_match_info, complete_stepnums, lc_patches_expected
):

    patches_with_missing_stepnums = dict()
    for patch in lc_patches_expected:
        try:
            avail_stepnums = patch_match_info[patch]
        except KeyError:
            avail_stepnums = []
        missing_stepnums = list(set(complete_stepnums) - set(avail_stepnums))
        missing_stepnums = sorted([int(x) for x in missing_stepnums])
        if len(missing_stepnums) > 0:
            patches_with_missing_stepnums[int(patch)] = missing_stepnums

    return patches_with_missing_stepnums


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("drn", help="Name of the directory storing cross-x files")
    parser.add_argument("z_min", help="Minimum expected redshift", type=float)
    parser.add_argument("z_max", help="Maximum expected reshift", type=float)
    parser.add_argument("-istart", help="First expected patch", type=int, default=0)
    parser.add_argument("-iend", help="Last expected patch", type=int, default=-1)
    parser.add_argument(
        "-lc_patch_list_cfg", help="fname to ASCII with list of sky patches", default=""
    )
    parser.add_argument(
        "-drn_cf", help="Name of the directory storing coreforest", default=DRN_LC_CORES
    )

    args = parser.parse_args()
    drn = args.drn
    z_min = args.z_min
    z_max = args.z_max

    lc_patch_list_cfg = args.lc_patch_list_cfg
    istart = args.istart
    iend = args.iend

    drn_cf = args.drn_cf

    if lc_patch_list_cfg == "":
        if iend == -1:
            _res = hlu.read_lc_ra_dec_patch_decomposition(FN_LC_CORES)
            patch_decomposition = _res[0]
            lc_patches_expected = patch_decomposition[:, 0].astype(int)
        else:
            lc_patches_expected = np.arange(istart, iend + 1).astype(int)
    else:
        lc_patches_expected = np.loadtxt(lc_patch_list_cfg).astype(int)

    complete_stepnums = hlu.get_timesteps_in_zrange(SIM_NAME, z_min, z_max)

    fn_pat = os.path.join(drn, BNPAT.format("*", "*"))
    bn_list = [os.path.basename(fn) for fn in glob(fn_pat)]
    patch_list = []
    step_list = []

    print("...getting step & patch of all available files")
    for bn in bn_list:
        stepnum, patch = get_stepnum_and_patch(bn)
        patch_list.append(patch)
        step_list.append(stepnum)

    uniq_patches = np.unique(patch_list)
    uniq_steps = np.unique(step_list)[::-1]

    patch_match_info = dict()
    step_match_info = dict()

    for patch in lc_patches_expected:
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

    stepnums_with_missing_patches = get_stepnums_with_missing_patches(
        step_match_info, complete_stepnums, lc_patches_expected
    )

    patches_with_missing_stepnums = get_patches_with_missing_stepnums(
        patch_match_info, complete_stepnums, lc_patches_expected
    )

    with open("stepnums_with_missing_patches.pickle", "wb") as handle:
        pickle.dump(
            stepnums_with_missing_patches, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    with open("patches_with_missing_stepnums.pickle", "wb") as handle:
        pickle.dump(
            patches_with_missing_stepnums, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    complete_stepnums = hlu.get_timesteps_in_zrange("LastJourney", 0, 3)
    final_step = complete_stepnums[-1]
    report = dict()
    for lc_patch, missing_stepnums in patches_with_missing_stepnums.items():
        if len(missing_stepnums) > 1:
            report[lc_patch] = missing_stepnums
        else:
            missing_step = missing_stepnums[0]
            if missing_step != final_step:
                report[lc_patch] = missing_stepnums
            else:
                bn = BNPAT.format(missing_step, lc_patch)
                bn = bn.replace("diffsky_data.hdf5", "hdf5")
                fn = os.path.join(drn_cf, bn)
                mock = load_flat_hdf5(fn, keys=["core_tag"])
                if len(mock["core_tag"]) > 0:
                    report[lc_patch] = missing_stepnums
