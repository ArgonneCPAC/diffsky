"""Script to make an SFH mock with DiffstarPop"""

import argparse
import os
from glob import glob
from time import time

import numpy as np
from diffmah.data_loaders.load_hacc_mahs import load_mahs_per_rank
from mpi4py import MPI

TMP_OUTPAT = "tmp_mah_fits_rank_{0}.dat"

LCRC_DRN_PAT = "/lcrc/group/cosmodata/simulations/DESI_W0WA/{0}/coreforest/forest"
POBOY_DRN_PAT = "/Users/aphearin/work/DATA/DESI_W0WA/{0}"

BNPAT = "m000p.coreforest.{}.hdf5"

NCHUNKS = 20
NUM_SUBVOLS_DISCOVERY = 96


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()
    parser = argparse.ArgumentParser()

    parser.add_argument("indir", help="Root directory storing both Discovery sims")
    parser.add_argument("sim_name", help="Simulation name", choices=["LCDM", "W0WA"])
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument(
        "-outbase", help="Basename of the output hdf5 file", default="diffmah_fits.h5"
    )
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument("-istart", help="First subvolume in loop", type=int, default=0)
    parser.add_argument(
        "-iend", help="Last subvolume in loop", type=int, default=NUM_SUBVOLS_DISCOVERY
    )
    parser.add_argument("-nstep", help="Number of gd steps", type=int, default=200)
    parser.add_argument("-nchunks", help="Number of chunks", type=int, default=NCHUNKS)
    parser.add_argument(
        "-num_subvols_tot",
        help="Total # subvols",
        type=int,
        default=NUM_SUBVOLS_DISCOVERY,
    )

    args = parser.parse_args()
    sim_name = args.sim_name
    nstep = args.nstep
    istart, iend = args.istart, args.iend

    num_subvols_tot = args.num_subvols_tot  # needed for string formatting
    outdir = args.outdir
    outbase = args.outbase
    nchunks = args.nchunks

    nchar_chunks = len(str(nchunks))

    os.makedirs(outdir, exist_ok=True)

    if args.indir == "POBOY":
        indir = POBOY_DRN_PAT.format(sim_name)
    elif args.indir == "LCRC":
        indir = LCRC_DRN_PAT.format(sim_name)
    else:
        indir = args.indir
    sim_name = "Discovery" + sim_name

    start = time()

    all_avail_subvol_fnames = glob(os.path.join(indir, "m000p.coreforest.*.hdf5"))
    all_avail_subvol_bnames = [os.path.basename(fn) for fn in all_avail_subvol_fnames]
    all_avail_subvolumes = [int(s.split(".")[2]) for s in all_avail_subvol_bnames]
    all_avail_subvolumes = np.array(sorted(all_avail_subvolumes))

    if args.test:
        subvolumes = [all_avail_subvolumes[0]]
        chunks = [0, 1, 2]
    else:
        subvolumes = all_avail_subvolumes[istart:iend]
        chunks = np.arange(nchunks).astype(int)

    for isubvol in subvolumes:
        isubvol_start = time()

        nchar_subvol = len(str(num_subvols_tot))

        subvol_str = f"{isubvol}"
        bname = BNPAT.format(subvol_str)
        fn_data = os.path.join(indir, bname)

        for chunknum in chunks:
            comm.Barrier()
            ichunk_start = time()

            tarr, mahs_for_rank = load_mahs_per_rank(
                fn_data, sim_name, chunknum, nchunks, comm=MPI.COMM_WORLD
            )
            nhalos_for_rank = mahs_for_rank.shape[0]
            nhalos_tot = comm.reduce(nhalos_for_rank, op=MPI.SUM)

            chunknum_str = f"{chunknum:0{nchar_chunks}d}"
            outbase_chunk = f"subvol_{subvol_str}_chunk_{chunknum_str}"
            rank_basepat = "_".join((outbase_chunk, TMP_OUTPAT))
            rank_outname = os.path.join(args.outdir, rank_basepat).format(rank)

            comm.Barrier()
            with open(rank_outname, "w") as fout:
                raise NotImplementedError()

            comm.Barrier()
            ichunk_end = time()
