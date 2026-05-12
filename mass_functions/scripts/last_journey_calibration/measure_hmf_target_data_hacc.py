""" """

import argparse
import gc
import os
from time import time

import numpy as np
from jax import random as jran

from diffsky.data_loaders.hacc_utils import get_diffsky_info_from_hacc_sim
from diffsky.data_loaders.hacc_utils import load_hacc_cores as lhc
from diffsky.mass_functions.measure_hmf import measure_cuml_hmf_target_data_counts

DRN_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney"
DRN_LJ_CORES_POBOY = os.path.join(DRN_LJ_POBOY, "coretrees")
DRN_LJ_DIFFMAH_POBOY = os.path.join(DRN_LJ_POBOY, "diffmah_fits")

DRN_SCRATCH_LCRC = "/lcrc/globalscratch/ahearin/"
DRN_SCRATCH_POBOY = "/Users/aphearin/work/DATA/random_data/0706"

BNPAT_CHUNK = "cuml_hmf_counts_subvol_{0}_chunk_{1}"
DEFAULT_NCHUNKS = 10

Z_TABLE = np.array((0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0))
LOGMP_BINS = np.linspace(11, 15.1, 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("sim_name", help="Name of HACC simulation")
    parser.add_argument("machine", help="Output directory", choices=["poboy", "lcrc"])
    parser.add_argument(
        "-outdir",
        help="Output directory to store target data. Default is scratch",
        default="",
    )
    parser.add_argument("-istart", help="First subvolume", type=int, default=0)
    parser.add_argument("-iend", help="Last subvolume", type=int, default=-1)
    parser.add_argument("-itest", help="Short test run?", type=int, default=0)
    parser.add_argument(
        "-nchunks",
        help="Number of chunks per subvol",
        type=int,
        default=DEFAULT_NCHUNKS,
    )

    args = parser.parse_args()
    sim_name = args.sim_name
    outdir = args.outdir
    machine = args.machine
    istart = args.istart
    iend = args.iend
    itest = args.itest
    nchunks = args.nchunks

    sim_info = get_diffsky_info_from_hacc_sim(sim_name)

    ran_key = jran.key(0)

    if machine == "lcrc":
        drn_cores = "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest"
        drn_diffmah = "/lcrc/project/halotools/LastJourney/diffmah_fits"
        drn_scratch = DRN_SCRATCH_LCRC
    elif machine == "poboy":
        drn_cores = DRN_LJ_CORES_POBOY
        drn_diffmah = DRN_LJ_DIFFMAH_POBOY
        drn_scratch = DRN_SCRATCH_POBOY

    if outdir == "":
        outdir = drn_scratch
    os.makedirs(outdir, exist_ok=True)

    if itest == 1:
        chunks = [0, 1]
        z_table = np.array((0.0, 1.0))
        istart, iend = 0, 1
    else:
        chunks = list(range(0, nchunks))
        z_table = Z_TABLE
        if iend == -1:
            iend = sim_info.num_subvols

    IZ_OBS = [np.argmin(np.abs(sim_info.z_sim - z)) for z in z_table]

    np.save(os.path.join(drn_scratch, "logmp_bins"), LOGMP_BINS)
    np.save(os.path.join(drn_scratch, "redshift_bins"), Z_TABLE)

    start = time()
    chunk_counter = 0
    for subvol in range(istart, iend):
        print(f"...Beginning to loop over chunks of subvolume {subvol}")

        for chunknum in chunks:
            ran_key, chunk_key = jran.split(ran_key, 2)

            collector_cens = []
            collector_all = []
            for iz in IZ_OBS:
                z_target = sim_info.z_sim[iz]
                args = (
                    sim_name,
                    subvol,
                    chunknum,
                    nchunks,
                    iz,
                    chunk_key,
                    drn_cores,
                    drn_diffmah,
                )

                diffsky_data = lhc.load_diffsky_data(*args)

                msk_cens = diffsky_data["subcat"].upids == -1

                logmp_data = diffsky_data["subcat"].logmp_t_obs
                counts_all = measure_cuml_hmf_target_data_counts(logmp_data, LOGMP_BINS)
                collector_all.append(counts_all)

                logmp_data = diffsky_data["subcat"].logmp_t_obs[msk_cens]
                counts_cens = measure_cuml_hmf_target_data_counts(
                    logmp_data, LOGMP_BINS
                )
                collector_cens.append(counts_cens)

            # centrals
            chunk_data_cens = np.array(collector_cens).T
            bname_chunk = BNPAT_CHUNK.format(subvol, chunknum) + "_cens"
            fn_out = os.path.join(drn_scratch, bname_chunk)
            np.save(fn_out, chunk_data_cens)

            # all
            chunk_data_all = np.array(collector_all).T
            bname_chunk = BNPAT_CHUNK.format(subvol, chunknum) + "_all"
            fn_out = os.path.join(drn_scratch, bname_chunk)
            np.save(fn_out, chunk_data_all)

            chunk_counter += 1
            del diffsky_data
            gc.collect()

    end = time()
    runtime = end - start
    msg = f"Runtime for {chunk_counter} total chunks = {runtime:.1f} seconds"
    print(msg)
