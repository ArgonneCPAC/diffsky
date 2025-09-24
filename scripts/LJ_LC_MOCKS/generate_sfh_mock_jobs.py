"""Script to generate scripts for production SFH mocks"""

import argparse
import os

import numpy as np

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
    parser.add_argument("drn_mock_out", help="Directory to write mock", default="")
    parser.add_argument(
        "-script_name",
        help="Filename of mock production script",
        default="make_sfh_lc_mock_lj.py",
    )
    parser.add_argument("-fn_patch_list", help="Filename of patch list", default="")
    parser.add_argument("-istart", help="Filename of patch list", default=-1, type=int)
    parser.add_argument("-iend", help="Filename of patch list", default=-1, type=int)

    parser.add_argument(
        "-drn_submit_script", help="Directory to write scripts", default=""
    )
    parser.add_argument(
        "-bn_submit_script",
        help="Basename of job submission script",
        default="run_sfh_mock_production.sh",
    )
    parser.add_argument(
        "-conda_env", help="conda environment to activate", default="improv311"
    )
    parser.add_argument(
        "-synthetic_cores",
        help="Use synthetic cores instead of simulated cores",
        default=0,
        type=int,
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
    drn_mock_out = args.drn_mock_out
    drn_submit_script = args.drn_submit_script
    bn_submit_script = args.bn_submit_script
    synthetic_cores = args.synthetic_cores

    if drn_submit_script == "":
        drn_submit_script = os.path.dirname(os.path.abspath(__file__))

    if fn_patch_list == "":
        patch_list = np.arange(istart, iend).astype(int)
    else:
        patch_list = np.loadtxt(fn_patch_list).astype(int)
    if len(patch_list) == 0:
        msg = f"fn_patch_list=`{fn_patch_list}` and (istart,iend)=({istart},{iend})"
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

    line_pat = "python {0} lcrc {1:.3f} {2:.3f} {3} {4} {5} "
    if synthetic_cores == 1:
        line_pat += "-synthetic_cores 1"

    bn_patch_list = os.path.basename(fn_patch_list)
    fn_submit_script = os.path.join(drn_submit_script, bn_submit_script)

    with open(fn_submit_script, "w") as fout:
        for line_out in header_lines:
            fout.write(line_out + "\n")

        for i in patch_list:
            j = i + 1
            line_out = line_pat.format(script_name, z_min, z_max, i, j, drn_mock_out)
            fout.write(line_out + "\n")
