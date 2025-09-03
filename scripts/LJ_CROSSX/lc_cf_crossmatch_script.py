"""Script to cross-match LastJourney lightcone and coreforest data"""

import argparse
import os
import pickle
from time import time

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from haccytrees import Simulation as HACCSim
from haccytrees import coretrees
from mpi4py import MPI

from diffsky.data_loaders import load_flat_hdf5
from diffsky.data_loaders.hacc_utils.defaults import DIFFMAH_MASS_COLNAME
from diffsky.data_loaders.hacc_utils.lightcone_utils import (
    LC_PATCH_BNPAT,
    collate_rank_data,
    get_diffsky_quantities_for_lc_patch,
    get_infall_times_lc_shell,
    get_lc_patches_in_zrange,
    initialize_lc_patch_data_out,
    load_lc_patch_data_out,
    overwrite_lc_patch_data_out,
)

DRN_LJ_CF_LCRC = "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest"
DRN_LJ_CF_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"

DRN_LJ_LC_LCRC = (
    "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/core-lc-5/output"
)
DRN_LJ_LC_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc_cores"

DRN_LJ_DIFFMAH_LCRC = "/lcrc/project/halotools/LastJourney/diffmah_fits"
DRN_LJ_DIFFMAH_POBOY = "/Users/aphearin/work/DATA/LastJourney/diffmah_fits"

DRN_LJ_CROSSX_OUT_LCRC = "/lcrc/project/halotools/LastJourney/lc-cf-diffsky"
DRN_LC_CF_XDATA_LCRC = os.path.join(DRN_LJ_CROSSX_OUT_LCRC, "LC_CF_XDATA")
DRN_LJ_CROSSX_OUT_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc-cf-diffsky"
DRN_LC_CF_XDATA_POBOY = os.path.join(DRN_LJ_CROSSX_OUT_POBOY, "LC_CF_XDATA")

DRN_LJ_CROSSX_OUT_LCRC_SCRATCH = "/lcrc/globalscratch/ahearin"

BNPAT_COREFOREST = "m000p.coreforest.{}.hdf5"
BNPAT_DIFFMAH_FITS = "subvol_{0}_diffmah_fits.hdf5"

NCHUNKS = 20
SIM_NAME = "LastJourney"

CF_FIELDS = ["central", "core_tag", "fof_halo_tag", "host_core", DIFFMAH_MASS_COLNAME]
shapes_1 = [f"infall_fof_halo_eigS1{x}" for x in ("X", "Y", "Z")]
shapes_2 = [f"infall_fof_halo_eigS2{x}" for x in ("X", "Y", "Z")]
shapes_3 = [f"infall_fof_halo_eigS3{x}" for x in ("X", "Y", "Z")]
core_data = ["merged", "vel_disp", "radius", "vx", "vy", "vz"]
CF_FIELDS = [*CF_FIELDS, *shapes_1, *shapes_2, *shapes_3, *core_data]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("z_min", help="Minimum redshift", type=float)
    parser.add_argument("z_max", help="Maximum redshift", type=float)

    parser.add_argument("-istart", help="First lc patch", type=int, default=0)
    parser.add_argument("-iend", help="Last lc patch", type=int, default=0)
    parser.add_argument(
        "-lc_patch_list_cfg", help="fname to ASCII with list of sky patches", default=""
    )
    parser.add_argument(
        "-machine", help="Machine nickname", choices=["lcrc", "poboy"], default="lcrc"
    )
    parser.add_argument("-itest", help="Test run?", default=0, type=int)
    parser.add_argument("-nchunks", help="Number of chunks", default=NCHUNKS, type=int)

    parser.add_argument(
        "-drn_out",
        help="Output drn of lightcone data",
        default=DRN_LJ_CROSSX_OUT_LCRC_SCRATCH,
    )

    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    args = parser.parse_args()
    z_min = args.z_min
    z_max = args.z_max
    machine = args.machine

    lc_patch_list_cfg = args.lc_patch_list_cfg
    istart = args.istart
    iend = args.iend

    itest = args.itest
    nchunks = args.nchunks
    drn_out = args.drn_out

    sim = HACCSim.simulations[SIM_NAME]

    if lc_patch_list_cfg == "":
        lc_patch_list = np.arange(istart, iend).astype(int)
    else:
        lc_patch_list = np.loadtxt(lc_patch_list_cfg).astype(int)

    if machine == "lcrc":
        drn_lc = DRN_LJ_LC_LCRC
        drn_cf = DRN_LJ_CF_LCRC
        drn_dmah = DRN_LJ_DIFFMAH_LCRC
        fn_cf_xdict = os.path.join(DRN_LC_CF_XDATA_LCRC, "cf_xdict.pickle")
        fn_lc_xdict = os.path.join(DRN_LC_CF_XDATA_LCRC, "lc_xdict.pickle")
        drn_out_scratch = DRN_LJ_CROSSX_OUT_LCRC_SCRATCH
    elif machine == "poboy":
        drn_lc = DRN_LJ_LC_POBOY
        drn_cf = DRN_LJ_CF_POBOY
        drn_dmah = DRN_LJ_DIFFMAH_LCRC
        fn_cf_xdict = os.path.join(DRN_LC_CF_XDATA_POBOY, "cf_xdict.pickle")
        fn_lc_xdict = os.path.join(DRN_LC_CF_XDATA_POBOY, "lc_xdict.pickle")
        drn_out_scratch = "DRN_TEMP"
        os.makedirs(DRN_LJ_CROSSX_OUT_LCRC_SCRATCH, exist_ok=True)
    else:
        raise ValueError(f"Unrecognized machine name = `{machine}`")

    os.makedirs(drn_out_scratch, exist_ok=True)

    with open(fn_cf_xdict, "rb") as handle:
        cf_xdict = pickle.load(handle)

    with open(fn_lc_xdict, "rb") as handle:
        lc_xdict = pickle.load(handle)

    lc_patches = get_lc_patches_in_zrange(
        SIM_NAME, lc_xdict, z_min, z_max, patch_list=lc_patch_list
    )

    subvolumes = []
    for patch_info in lc_patches:
        subvolumes_for_patch = lc_xdict[patch_info]
        subvolumes.extend(subvolumes_for_patch)
    all_overlapping_subvolumes = np.unique(subvolumes)
    if rank == 0:
        print(f"Overlapping subvolumes = {all_overlapping_subvolumes}")

    if itest == 1:
        all_overlapping_subvolumes = [all_overlapping_subvolumes[0]]
        all_chunks = [0, 1]
    else:
        all_chunks = np.arange(NCHUNKS).astype(int)

    chunks_for_rank = np.array_split(all_chunks, nranks)[rank]

    start_script = time()
    for isubvol in all_overlapping_subvolumes:

        start_subvol = time()
        if rank == 0:
            print(f"\n...working on subvolume {isubvol}")

        bname_coretree = BNPAT_COREFOREST.format(isubvol)
        fname_coretree = os.path.join(drn_cf, bname_coretree)
        olap_patch_ids = list(set(cf_xdict[isubvol]) & set(lc_patches))

        shells_subvol = np.unique([x[0] for x in olap_patch_ids])

        bname_dmah = BNPAT_DIFFMAH_FITS.format(isubvol)
        fname_dmah = os.path.join(drn_dmah, bname_dmah)

        for ichunk in chunks_for_rank:
            comm.Barrier()
            start_chunk = time()

            forest_matrices = coretrees.corematrix_reader(
                fname_coretree,
                calculate_secondary_host_row=True,
                nchunks=nchunks,
                chunknum=ichunk,
                simulation="LastJourney",
                include_fields=CF_FIELDS,
            )

            cf_first_row = forest_matrices["absolute_row_idx"][0]
            cf_last_row = forest_matrices["absolute_row_idx"][-1]
            if ichunk < nchunks - 1:
                diffmah_data = load_flat_hdf5(
                    fname_dmah, istart=cf_first_row, iend=cf_last_row + 1
                )
            else:
                diffmah_data = load_flat_hdf5(fname_dmah, istart=cf_first_row)

            _some_diffmah_key = DEFAULT_MAH_PARAMS._fields[0]
            n_diffmah = len(diffmah_data[_some_diffmah_key])
            n_cf = len(forest_matrices["absolute_row_idx"])
            assert n_diffmah == n_cf, "mismatch between diffmah fits and coreforest"

            for ishell in shells_subvol:
                olap_patch_ids_ishell = [
                    x
                    for x in olap_patch_ids
                    if (x[0] == ishell) and (x[1] in lc_patch_list)
                ]

                timestep_idx = np.searchsorted(sim.cosmotools_steps, ishell)
                _res = get_infall_times_lc_shell(forest_matrices, timestep_idx)
                cf_indx_t_ult_inf, cf_indx_t_pen_inf = _res

                for olap_patch_id in olap_patch_ids_ishell:
                    ishell, ipatch = olap_patch_id
                    bn_patch_in = LC_PATCH_BNPAT.format(ishell, ipatch)
                    fn_patch_in = os.path.join(drn_lc, bn_patch_in)
                    lc_patch_data = load_flat_hdf5(fn_patch_in)
                    n_patch = len(lc_patch_data["coreforest_file_idx"])

                    try:
                        lc_patch_data_out = load_lc_patch_data_out(
                            drn_out_scratch, bn_patch_in, rank
                        )
                    except FileNotFoundError:
                        lc_patch_data_out = initialize_lc_patch_data_out(n_patch)

                    lc_patch_data_out = get_diffsky_quantities_for_lc_patch(
                        lc_patch_data,
                        lc_patch_data_out,
                        forest_matrices,
                        diffmah_data,
                        isubvol,
                        cf_indx_t_ult_inf,
                        cf_indx_t_pen_inf,
                        timestep_idx,
                    )
                    overwrite_lc_patch_data_out(
                        lc_patch_data_out, drn_out_scratch, bn_patch_in, rank
                    )

            end_chunk = time()
            runtime_chunk = end_chunk - start_chunk

        end_subvol = time()
        runtime_subvol = end_subvol - start_subvol
        if rank == 0:
            print(f"Runtime for subvolume {isubvol} = {runtime_subvol:.2f} seconds\n")

    end_script = time()
    if rank == 0:
        collate_rank_data(drn_out_scratch, drn_out, lc_patches, nranks)

    if rank == 0:
        runtime_script = (end_script - start_script) / 60.0
        if rank == 0:
            print(f"Runtime for script = {runtime_script:.2f} minutes\n")
