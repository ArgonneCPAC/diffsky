"""Script to make an SFH mock with DiffstarPop"""

import argparse
import os
import subprocess
from time import time

import numpy as np
from diffstarpop import DEFAULT_DIFFSTARPOP_PARAMS
from diffstarpop.mc_diffstarpop_cen_tpeak import mc_diffstar_sfh_galpop_cen
from diffstarpop.param_utils import mc_select_diffstar_params
from haccytrees import Simulation as HACCSim
from jax import random as jran
from mpi4py import MPI

from diffsky.data_loaders import hacc_core_utils as hcu
from diffsky.data_loaders import load_hacc_cores as lhc

OUTPAT_CHUNK_RANK = "sfh_mock_subvol_{0}_chunk_{1}_rank_{2}.hdf5"
OUTPAT_CHUNK = "sfh_mock_subvol_{0}_chunk_{1}.hdf5"
OUTPAT_SUBVOL = "sfh_mock_subvol_{0}.hdf5"

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
    parser.add_argument(
        "-redshift", help="redshift of output mock", type=float, default=0.0
    )
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
    redshift = args.redshift
    indir_cores = args.indir_cores
    indir_diffmah = args.indir_diffmah
    sim_name = args.sim_name
    machine = args.machine
    istart, iend = args.istart, args.iend

    num_subvols_tot = args.num_subvols_tot  # needed for string formatting
    outdir = args.outdir
    outbase = args.outbase
    nchunks = args.nchunks

    sim = HACCSim.simulations[sim_name]
    zarr_sim = sim.step2z(np.array(sim.cosmotools_steps))
    iz_obs = np.argmin(np.abs(redshift - zarr_sim))

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

            diffsky_data = lhc.load_diffsky_data_per_rank(
                sim_name,
                isubvol,
                chunknum,
                nchunks,
                iz_obs,
                chunk_key_for_rank,
                indir_cores,
                indir_diffmah,
                comm=MPI.COMM_WORLD,
            )
            fb, lgt0 = hcu.get_diffstar_cosmo_quantities(sim_name)

            comm.Barrier()
            args = (
                DEFAULT_DIFFSTARPOP_PARAMS,
                diffsky_data["subcat"].mah_params,
                diffsky_data["subcat"].logmp0,
                chunk_key_for_rank,
                diffsky_data["tarr"],
            )
            _res = mc_diffstar_sfh_galpop_cen(*args, lgt0=lgt0, fb=fb)
            diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

            sfh_params = mc_select_diffstar_params(
                diffstar_params_q, diffstar_params_ms, mc_is_q
            )

            chunknum_str = f"{chunknum:0{nchar_chunks}d}"
            bname = OUTPAT_CHUNK_RANK.format(subvol_str, chunknum_str, rank)
            rank_outname = os.path.join(outdir, bname)

            lhc.write_sfh_mock_to_disk(diffsky_data, sfh_params, rank_outname)

            comm.Barrier()
            ichunk_end = time()

            if rank == 0:
                chunk_fnames = []
                for irank in range(nranks):
                    bname = OUTPAT_CHUNK_RANK.format(subvol_str, chunknum_str, irank)
                    fname = os.path.join(outdir, bname)
                    chunk_fnames.append(fname)
                bnout = OUTPAT_CHUNK.format(subvol_str, chunknum_str)
                fnout = os.path.join(outdir, bnout)
                lhc.collate_hdf5_file_collection(chunk_fnames, fnout)
                bnpat = OUTPAT_CHUNK_RANK.format(subvol_str, chunknum_str, "*")
                fnpat = os.path.join(outdir, bnpat)
                command = "rm -rf " + fnpat
                raw_result = subprocess.check_output(command, shell=True)

            comm.Barrier()
