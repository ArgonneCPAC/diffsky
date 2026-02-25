"""Script used by mock_generation_test.yaml workflow to dummy up a complete set of
mock galaxies from synthetics halos only. Used to incorporate OpenCosmo into our CI"""

import argparse
import os
import subprocess
from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("drn", help="Location of mock data")

    args = parser.parse_args()

    fn_list = glob(os.path.join(args.drn, "*diffsky_gals.hdf5"))
    assert len(fn_list) > 0
    for fn in fn_list:
        bn = os.path.basename(fn)
        bn_new = bn.replace("diffsky_gals.synthetic_halos.hdf5", "diffsky_gals.hdf5")
        fn_new = os.path.join(args.drn, bn_new)
        command = f"cp {fn} {fn_new}"
        raw_result = subprocess.check_output(command, shell=True)
