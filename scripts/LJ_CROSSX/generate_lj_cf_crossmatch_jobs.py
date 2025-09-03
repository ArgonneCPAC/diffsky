"""Script to generate scripts for cross-matching LastJourney core lightcones

python generate_lj_cf_crossmatch_jobs.py galsampler 4 0.01 3.0 /Users/aphearin/work/random/0903/CROSSX_JOBS -istart 0 -iend 5

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
    parser.add_argument("--dry_run", action="store_false", help="Submit jobs")

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
    dry_run = args.dry_run

    if drn_submit_script == "":
        drn_submit_script = os.path.dirname(os.path.abspath(__file__))

    if fn_patch_list == "":
        patch_list = np.arange(istart, iend).astype(int)
    else:
        patch_list = np.loadtxt(fn_patch_list).astype(int)
    if len(patch_list) == 0:
        msg = f"fn_patch_list='{fn_patch_list}' and (istart,iend)=({istart},{iend})"
        raise ValueError(msg)

    header_lines = (
        "#!/bin/bash",
        "",
        "# join error into standard out file <job_name>.o<job_id>",
        "# PBS -j oe",
        "",
        "# account to charge",
        f"#PBS -A {account}",
        "",
        "# allocate {select} nodes, each with {mpiprocs} MPI processes",
        "#PBS -l select=1:mpiprocs=1",
        "",
        f"#PBS -l walltime={walltime}:00:00",
        "",
        "# Load software",
        "source ~/.bash_profile",
        f"conda activate {conda_env}",
        "",
        f"cd {drn_submit_script}",
        "",
    )

    line_pat = "python {0} {1:.3f} {2:.3f} {3} {4} -drn_out {5} "
    # python lc_cf_crossmatch_script.py 0.01 3.0 -istart 0 -iend 5 -drn_out /lcrc/project/halotools/random_data/0826

    bn_patch_list = os.path.basename(fn_patch_list)
    fn_submit_script = os.path.join(drn_submit_script, BN_JOB.format(istart, iend))

    with open(fn_submit_script, "w") as fout:
        for line_out in header_lines:
            fout.write(line_out + "\n")

        line_out = line_pat.format(script_name, z_min, z_max, istart, iend, drn_out)
        fout.write(line_out + "\n")
