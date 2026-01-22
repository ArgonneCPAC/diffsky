""""""

import argparse
import os
from glob import glob

import numpy as np

BN_PAT = "lc_cores-{0}.{1}.*.hdf5"


def get_stepnum_and_patch(bname):
    _stepnum, _patch = bname.split("-")[1].split(".")[:2]
    stepnum = int(_stepnum)
    patch = int(_patch)
    return stepnum, patch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("drn", help="Name of the directory storing cross-x files")
    parser.add_argument("z_min", help="Minimum expected redshift", type=float)
    parser.add_argument("z_max", help="Maximum expected reshift", type=float)

    args = parser.parse_args()
    drn = args.drn
    z_min = args.z_min
    z_max = args.z_max

    fn_pat = os.path.join(drn, BN_PAT.format("*", "*"))
    fn_list = glob(fn_pat)
    bn_list = [os.path.basename(fn) for fn in fn_list]

    step_collector = np.unique([get_stepnum_and_patch(bn)[0] for bn in bn_list])
    patch_collector = np.unique([get_stepnum_and_patch(bn)[1] for bn in bn_list])
