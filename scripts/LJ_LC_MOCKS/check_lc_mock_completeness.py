""""""

import argparse
import os
from glob import glob

BN_PAT = "lc_cores-{0}.{1}.*.hdf5"

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
