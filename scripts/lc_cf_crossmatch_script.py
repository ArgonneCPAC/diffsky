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
from diffsky.data_loaders.hacc_utils import lightcone_utils as hlu
from diffsky.data_loaders.hacc_utils.defaults import DIFFMAH_MASS_COLNAME

DRN_LJ_CF_LCRC = "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest"
DRN_LJ_CF_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"

DRN_LJ_LC_LCRC = "/lcrc/project/cosmo_ai/mbuehlmann/LastJourney/core-lc-3/output"
DRN_LJ_LC_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc_cores"

DRN_LJ_DIFFMAH_LCRC = "/lcrc/project/halotools/LastJourney/diffmah_fits"
DRN_LJ_DIFFMAH_POBOY = "/Users/aphearin/work/DATA/LastJourney/diffmah_fits"

DRN_LJ_CROSSX_OUT_LCRC = "/lcrc/project/halotools/LastJourney/lc-cf-diffsky"
DRN_LC_CF_XDATA_LCRC = os.path.join(DRN_LJ_CROSSX_OUT_LCRC, "LC_CF_XDATA")
DRN_LJ_CROSSX_OUT_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc-cf-diffsky"
DRN_LC_CF_XDATA_POBOY = os.path.join(DRN_LJ_CROSSX_OUT_POBOY, "LC_CF_XDATA")

DRN_LJ_CROSSX_OUT_LCRC_SCRATCH = "/lcrc/globalscratch/ahearin"

BNPAT_COREFOREST = "m000p.coreforest.{}.hdf5"
BNPAT_LC_CORES = "lc_cores-{0}.{1}.hdf5"
BNPAT_DIFFMAH_FITS = "subvol_{0}_diffmah_fits.hdf5"

NCHUNKS = 20
NUM_SUBVOLS_LJ_CF = 192
SIM_NAME = "LastJourney"

CF_FIELDS = ["central", "core_tag", "fof_halo_tag", "host_core", DIFFMAH_MASS_COLNAME]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "lc_patch_list_cfg", help="fname to ASCII with list of sky patches"
    )
    parser.add_argument("z_min", help="Minimum redshift", type=float)
    parser.add_argument("z_max", help="Maximum redshift", type=float)

    parser.add_argument(
        "-machine", help="Machine nickname", choices=["lcrc", "poboy"], default="lcrc"
    )
    parser.add_argument("-itest", help="Test run?", default=0, type=int)
    parser.add_argument("-nchunks", help="Number of chunks", default=NCHUNKS, type=int)

    parser.add_argument(
        "-drn_out",
        help="Output directory of lightcone data",
        default=DRN_LJ_CROSSX_OUT_LCRC_SCRATCH,
    )

    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    args = parser.parse_args()
    z_min = args.z_min
    z_max = args.z_max
    lc_patch_list_cfg = args.lc_patch_list_cfg

    machine = args.machine
    itest = args.itest
    nchunks = args.nchunks
    drn_out = args.drn_out

    sim = HACCSim.simulations[SIM_NAME]

    lc_patch_list = np.loadtxt(lc_patch_list_cfg).astype(int)

    if machine == "lcrc":
        drn_lc = DRN_LJ_LC_LCRC
        drn_cf = DRN_LJ_CF_LCRC
        drn_dmah = DRN_LJ_DIFFMAH_LCRC
        fn_cf_xdict = os.path.join(DRN_LC_CF_XDATA_LCRC, "cf_xdict.pickle")
        fn_lc_xdict = os.path.join(DRN_LC_CF_XDATA_LCRC, "lc_xdict.pickle")
        os.makedirs(DRN_LJ_CROSSX_OUT_LCRC_SCRATCH, exist_ok=True)
    elif machine == "poboy":
        drn_lc = DRN_LJ_LC_POBOY
        drn_cf = DRN_LJ_CF_POBOY
        drn_dmah = DRN_LJ_DIFFMAH_LCRC
        fn_cf_xdict = os.path.join(DRN_LC_CF_XDATA_POBOY, "cf_xdict.pickle")
        fn_lc_xdict = os.path.join(DRN_LC_CF_XDATA_POBOY, "lc_xdict.pickle")
    else:
        raise ValueError(f"Unrecognized machine name = `{machine}`")

    with open(fn_cf_xdict, "rb") as handle:
        cf_xdict = pickle.load(handle)

    with open(fn_lc_xdict, "rb") as handle:
        lc_xdict = pickle.load(handle)

    lc_patches = hlu.get_lc_patches_in_zrange(
        SIM_NAME, lc_xdict, z_min, z_max, patch_list=lc_patch_list
    )

    subvolumes = []
    for patch_info in lc_patches:
        subvolumes_for_patch = lc_xdict[patch_info]
        subvolumes.extend(subvolumes_for_patch)
    all_overlapping_subvolumes = np.unique(subvolumes)
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
            print(f"...working on chunk {ichunk}")

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
                _res = hlu.get_infall_times_lc_shell(forest_matrices, timestep_idx)
                cf_indx_t_ult_inf, cf_indx_t_pen_inf = _res

                for olap_patch_id in olap_patch_ids_ishell:
                    ishell, ipatch = olap_patch_id
                    bn_patch_in = BNPAT_LC_CORES.format(ishell, ipatch)
                    fn_patch_in = os.path.join(drn_lc, bn_patch_in)
                    lc_patch_data = load_flat_hdf5(fn_patch_in)
                    n_patch = len(lc_patch_data["file_idx"])

                    try:
                        lc_patch_data_out = hlu.load_lc_patch_data_out(
                            drn_out, bn_patch_in, rank
                        )
                    except FileNotFoundError:
                        lc_patch_data_out = hlu.initialize_lc_patch_data_out(n_patch)

                    lc_patch_data_out = hlu.get_diffsky_quantities_for_lc_patch(
                        lc_patch_data,
                        lc_patch_data_out,
                        forest_matrices,
                        diffmah_data,
                        isubvol,
                        cf_indx_t_ult_inf,
                        cf_indx_t_pen_inf,
                        timestep_idx,
                    )
                    hlu.overwrite_lc_patch_data_out(
                        lc_patch_data_out, drn_out, bn_patch_in, rank
                    )

            end_chunk = time()
            runtime_chunk = end_chunk - start_chunk
            print(f"Runtime for chunk {ichunk} = {runtime_chunk:.2f} seconds\n")

        end_subvol = time()
        runtime_subvol = end_subvol - start_subvol
        print(f"Runtime for subvolume {isubvol} = {runtime_subvol:.2f} seconds\n")

    end_script = time()
    runtime_script = end_script - start_script
    print(f"Runtime for script = {runtime_script:.2f} seconds\n")
