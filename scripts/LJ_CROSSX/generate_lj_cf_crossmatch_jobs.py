"""Script to generate scripts for cross-matching LastJourney core lightcones

python generate_lj_cf_crossmatch_jobs.py galsampler 8 0.01 3.0 /lcrc/project/halotools/random_data/0903 -istart 0 -iend 20 -drn_script /home/ahearin/work/random/0903/CROSSX_JOBS

"""

import argparse
import os
import subprocess

import numpy as np

BN_JOB = "run_lc_cf_crossx_{0}_to_{1}.sh"
BN_SCRIPT = "lc_cf_crossmatch_script.py"

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
    parser.add_argument("-fn_patch_list", help="Filename of patch list", default="")
    parser.add_argument("-istart", help="Filename of patch list", default=-1, type=int)
    parser.add_argument("-iend", help="Filename of patch list", default=-1, type=int)
    parser.add_argument("--submit_job", action="store_true", help="Submit jobs")
    parser.add_argument(
        "-job_size", help="Number of patches per job", default=5, type=int
    )

    parser.add_argument("-drn_script", help="Directory to write scripts", default="")

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
    job_size = args.job_size
    conda_env = args.conda_env
    drn_out = args.drn_out
    drn_script = args.drn_script
    submit_job = args.submit_job

    if drn_script == "":
        drn_script = os.path.dirname(os.path.abspath(__file__))

    if fn_patch_list == "":
        patch_list = np.arange(istart, iend).astype(int)
    else:
        patch_list = np.loadtxt(fn_patch_list).astype(int)
    if len(patch_list) == 0:
        msg = f"fn_patch_list='{fn_patch_list}' and (istart,iend)=({istart},{iend})"
        raise ValueError(msg)

    n_jobs = len(patch_list) // job_size
    job_list = np.array_split(patch_list, n_jobs)
    print(f"\nCreating {n_jobs} jobs")
    node_hours_tot = walltime * n_jobs
    print(f"Total node hours = {node_hours_tot:.1f}\n")

    nchar = len(str(np.max(patch_list)))

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
        f"cd {drn_script}",
        "",
        f"rsync /home/ahearin/work/repositories/python/diffsky/scripts/LJ_CROSSX/{BN_SCRIPT} ./",
        "",
    )

    line_pat = "python {0} {1:.3f} {2:.3f} -istart {3} -iend {4} -drn_out {5} "

    if submit_job:
        print("Submitting jobs to queue")
    else:
        print("Dry run: jobs not submitted")

    for job_info in job_list:

        i = job_info[0]
        j = job_info[-1]
        ibn = f"{i:0{nchar}d}"
        jbn = f"{j:0{nchar}d}"

        fn_submit_script = os.path.join(drn_script, BN_JOB.format(ibn, jbn))

        with open(fn_submit_script, "w") as fout:
            for line_out in header_lines:
                fout.write(line_out + "\n")

            line_out = line_pat.format(BN_SCRIPT, z_min, z_max, i, j, drn_out)
            fout.write(line_out + "\n")

        if submit_job:
            command = f"qsub {fn_submit_script}"
            raw_result = subprocess.check_output(command, shell=True)
