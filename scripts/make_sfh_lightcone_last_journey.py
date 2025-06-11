"""Script to make an SFH mock using DiffstarPop to populate a Last Journey lightcone"""

import argparse
import os
import pickle
from time import time

import numpy as np
from jax import random as jran
from mpi4py import MPI

from diffsky.data_loaders import mpi_utils
from diffsky.data_loaders.hacc_utils import lc_mock_production as lcmp
from diffsky.data_loaders.hacc_utils import lightcone_utils as hlu
from diffsky.data_loaders.hacc_utils import load_lc_cf

DRN_LJ_CF_LCRC = "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest"
DRN_LJ_CF_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"

DRN_LJ_LC_LCRC = "/lcrc/project/cosmo_ai/mbuehlmann/LastJourney/core-lc-3/output"
DRN_LJ_LC_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc_cores"

DRN_LJ_CROSSX_OUT_LCRC = "/lcrc/project/halotools/LastJourney/lc-cf-diffsky"
DRN_LC_CF_XDATA_LCRC = os.path.join(DRN_LJ_CROSSX_OUT_LCRC, "LC_CF_XDATA")
DRN_LJ_CROSSX_OUT_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc-cf-diffsky"
DRN_LC_CF_XDATA_POBOY = os.path.join(DRN_LJ_CROSSX_OUT_POBOY, "LC_CF_XDATA")

LC_XDICT_BNAME = "lc_xdict.pickle"

LC_CF_BNPAT = "lc_cores-{0}.{1}.diffsky_data.hdf5"
SIM_NAME = "LastJourney"
BNPAT_OUT = "diffsky_{0}.{1}.hdf5"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "machine", help="Machine name where script is run", choices=["lcrc", "poboy"]
    )
    parser.add_argument(
        "lc_patch_list_cfg", help="fname to ASCII with list of sky patches"
    )
    parser.add_argument("z_min", help="Minimum redshift", type=float)
    parser.add_argument("z_max", help="Maximum redshift", type=float)

    parser.add_argument("drn_out", help="Output directory")
    parser.add_argument(
        "-indir_lc_data",
        help="Input drn storing lc_cores-*.*.hdf5",
        default=DRN_LJ_LC_LCRC,
    )
    parser.add_argument(
        "-bnpat_out", help="Basename pattern of output file", default=BNPAT_OUT
    )

    comm = MPI.COMM_WORLD
    rankinfo, sorted_infolist = mpi_utils.get_mpi_rank_info(comm)
    rank, intra_node_id, node_number = rankinfo
    nranks_tot = comm.Get_size()
    n_nodes_tot = len(set([x[2] for x in sorted_infolist]))
    nranks_per_node = len(set([x[1] for x in sorted_infolist]))

    args = parser.parse_args()
    machine = args.machine
    lc_patch_list_cfg = args.lc_patch_list_cfg
    z_min = args.z_min
    z_max = args.z_max
    drn_out = args.drn_out
    bnpat_out = args.bnpat_out

    if machine == "poboy":
        indir_lc_diffsky = DRN_LJ_CROSSX_OUT_POBOY
        indir_lc_data = DRN_LJ_CROSSX_OUT_POBOY
        fn_lc_xdict = os.path.join(DRN_LC_CF_XDATA_POBOY, LC_XDICT_BNAME)
    elif machine == "lcrc":
        indir_lc_diffsky = DRN_LJ_CROSSX_OUT_LCRC
        indir_lc_data = DRN_LJ_LC_LCRC
        fn_lc_xdict = os.path.join(DRN_LC_CF_XDATA_LCRC, LC_XDICT_BNAME)

    with open(fn_lc_xdict, "rb") as handle:
        lc_xdict = pickle.load(handle)

    ran_key = jran.key(0)

    sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim(SIM_NAME)

    lc_patch_list = np.atleast_1d(np.loadtxt(lc_patch_list_cfg).astype(int))
    lc_patch_list_for_node = np.array_split(lc_patch_list, n_nodes_tot)[node_number]

    start_script = time()
    for lc_patch in lc_patch_list_for_node:
        ran_key, patch_key, shuffle_key = jran.split(ran_key, 3)

        lc_patch_info_list = sorted(
            hlu.get_lc_patches_in_zrange(
                SIM_NAME, lc_xdict, z_min, z_max, patch_list=[lc_patch]
            )
        )
        fn_list_lc_patch = [
            os.path.join(indir_lc_diffsky, LC_CF_BNPAT.format(*patch_info))
            for patch_info in lc_patch_info_list
        ]

        if rank == 0:
            print(f"\n{len(fn_list_lc_patch)} timestep files for lc_patch = {lc_patch}")

        indx_all = np.arange(len(lc_patch_info_list)).astype(int)
        indx_all = jran.permutation(shuffle_key, indx_all)
        indx_rank = np.array_split(indx_all, nranks_per_node)[intra_node_id]

        start = time()
        for indx in indx_rank:
            fn_lc_diffsky = fn_list_lc_patch[indx]

            lc_data, diffsky_data = load_lc_cf.load_lc_diffsky_patch_data(
                fn_lc_diffsky, indir_lc_data
            )

            patch_key, fn_key = jran.split(patch_key, 2)
            lc_data, diffsky_data = lcmp.add_sfh_quantities_to_mock(
                sim_info, lc_data, diffsky_data, fn_key
            )

            bn_lc_diffsky = os.path.basename(fn_lc_diffsky)
            bn_out = bn_lc_diffsky.replace("diffsky_data", "diffsky_gals")
            fn_out = os.path.join(drn_out, bn_out)
            lcmp.write_lc_sfh_mock_to_disk(fn_out, lc_data, diffsky_data)

        end = time()
        runtime = (end - start) / 60.0
        if rank == 0:
            print(f"Runtime to product lc_patch {lc_patch} = {runtime:.1f} minutes\n")

    end_script = time()
    n_patches = len(lc_patch_list)
    runtime = end_script - start_script
    msg = f"Total runtime for {n_patches} patches = {runtime:.1f} minutes"
