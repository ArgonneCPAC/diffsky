"""Script to generate scripts for cross-matching LastJourney core lightcones

python generate_lj_cf_crossmatch_jobs.py galsampler 4 0.01 3.0 /Users/aphearin/work/random/0903/CROSSX_JOBS

"""

import argparse
import os

import numpy as np

BN_JOB = "run_lc_cf_crossmatch_{0}_to_{1}.sh"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "account",
        help="Account to charge hours",
        choices=["halotools", "galsampler", "cosmo_ai"],
    )
    parser.add_argument("walltime", help="hours to charge", type=int)
    parser.add_argument("z_min", help="Minimum redshift", type=float)
    parser.add_argument("z_max", help="Maximum redshift", type=float)
    parser.add_argument("drn_out", help="Directory to write mock", default="")
    parser.add_argument(
        "-script_name",
        help="Filename of cross-matching script",
        default="lc_cf_crossmatch_script.py",
    )
    parser.add_argument("-fn_patch_list", help="Filename of patch list", default="")
    parser.add_argument("-istart", help="Filename of patch list", default=-1, type=int)
    parser.add_argument("-iend", help="Filename of patch list", default=-1, type=int)

    parser.add_argument(
        "-drn_submit_script", help="Directory to write scripts", default=""
    )

    parser.add_argument(
        "-conda_env", help="conda environment to activate", default="improv311"
    )

    args = parser.parse_args()
    account = args.account
    walltime = args.walltime
    z_min = args.z_min
    z_max = args.z_max
    fn_patch_list = args.fn_patch_list
    istart = args.istart
    iend = args.iend
    conda_env = args.conda_env
    script_name = args.script_name
    drn_out = args.drn_out
    drn_submit_script = args.drn_submit_script
    bn_submit_script = args.bn_submit_script
