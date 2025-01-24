"""Script to make an SFH mock with DiffstarPop"""

import argparse
import os
from time import time

import numpy as np
from diffmah.data_loaders.load_hacc_mahs import load_mahs_per_rank
from jax import random as jran
from mpi4py import MPI

from diffsky.data_loaders import load_hacc_cores as lhc

TMP_OUTPAT = "tmp_mah_fits_rank_{0}.dat"

DRN_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"
DRN_LJ_LCRC = "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest"

DRN_LJ_DMAH_POBOY = "/Users/aphearin/work/DATA/LastJourney/diffmah_fits"
DRN_LJ_DMAH_LCRC = "/lcrc/project/halotools/LastJourney/diffmah_fits"

BNPAT_CORE_DATA = "m000p.coreforest.{}.hdf5"

NCHUNKS = 20
NUM_SUBVOLS_LJ = 192


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()
    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("-indir_cores", help="Drn of HACC core data", default=None)
    parser.add_argument("-indir_diffmah", help="Drn of diffmah data", default=None)
    parser.add_argument("-sim_name", help="Simulation name", default="LastJourney")
    parser.add_argument(
        "-machine",
        help="Machine name",
        default="poboy",
        type=str,
        choices=["lcrc", "poboy"],
    )
    parser.add_argument(
        "-outbase", help="Basename of the output hdf5 file", default="sfh_mock.hdf5"
    )
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument("-istart", help="First subvolume in loop", type=int, default=0)
    parser.add_argument(
        "-iend", help="Last subvolume in loop", type=int, default=NUM_SUBVOLS_LJ
    )
    parser.add_argument("-nchunks", help="Number of chunks", type=int, default=NCHUNKS)
    parser.add_argument(
        "-num_subvols_tot", help="Total # subvols", type=int, default=NUM_SUBVOLS_LJ
    )

    args = parser.parse_args()
    indir_cores = args.indir_cores
    indir_diffmah = args.indir_diffmah
    sim_name = args.sim_name
    machine = args.machine
    istart, iend = args.istart, args.iend

    num_subvols_tot = args.num_subvols_tot  # needed for string formatting
    outdir = args.outdir
    outbase = args.outbase
    nchunks = args.nchunks

    nchar_chunks = len(str(nchunks))

    os.makedirs(outdir, exist_ok=True)

    rank_key = jran.key(rank)

    if args.machine == "poboy":
        indir_cores = DRN_LJ_POBOY
        indir_diffmah = DRN_LJ_DMAH_POBOY
    elif args.machine == "lcrc":
        indir_cores = DRN_LJ_LCRC
        indir_diffmah = DRN_LJ_DMAH_LCRC
    else:
        raise ValueError("Unrecognized machine name")

    start = time()

    if args.test:
        subvolumes = [0]
        chunks = [0, 1]
    else:
        subvolumes = np.arange(istart, iend + 1).astype(int)
        chunks = np.arange(nchunks).astype(int)
    subvolumes = sorted(subvolumes)

    all_avail_subvol_fnames = lhc._get_all_avail_basenames(
        indir_cores, BNPAT_CORE_DATA, subvolumes
    )

    for isubvol in subvolumes:
        isubvol_start = time()

        subvol_str = f"{isubvol}"
        bname_core_data = BNPAT_CORE_DATA.format(subvol_str)
        fn_data = os.path.join(indir_cores, bname_core_data)

        for chunknum in chunks:
            comm.Barrier()
            rank_key, chunk_key_for_rank = jran.split(rank_key, 2)
            ichunk_start = time()

            tarr, mahs_for_rank = load_mahs_per_rank(
                fn_data, sim_name, chunknum, nchunks, comm=MPI.COMM_WORLD
            )
            nhalos_for_rank = mahs_for_rank.shape[0]
            nhalos_tot = comm.reduce(nhalos_for_rank, op=MPI.SUM)

            load_discovery_diffsky_data(
                sim_name,
                isubvol,
                chunknum,
                nchunks,
                iz_obs,
                chunk_key_for_rank,
                drn_cores=indir_cores,
                drn_diffmah=indir_diffmah,
            )

            chunknum_str = f"{chunknum:0{nchar_chunks}d}"
            outbase_chunk = f"subvol_{subvol_str}_chunk_{chunknum_str}"
            rank_basepat = "_".join((outbase_chunk, TMP_OUTPAT))
            rank_outname = os.path.join(args.outdir, rank_basepat).format(rank)
            raise NotImplementedError()

            comm.Barrier()
            with open(rank_outname, "w") as fout:
                raise NotImplementedError()

            comm.Barrier()
            ichunk_end = time()
