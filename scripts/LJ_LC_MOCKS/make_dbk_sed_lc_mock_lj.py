"""Script to make an SED mock to populate a Last Journey lightcone

To run a unit test of this script:

python scripts/LJ_LC_MOCKS/make_dbk_sed_lc_mock_lj.py  poboy 0.2 0.21 0 1 ci_test_output ci_test_mock -fn_u_params sfh_model -sfh_model smdpl_dr1 -synthetic_cores 1 -lgmp_min 12.0 -lgmp_max 13.0

python scripts/LJ_LC_MOCKS/inspect_lc_mock.py ci_test_output/synthetic_cores/smdpl_dr1

"""  # noqa

import argparse
import gc
import os
import sys
from time import sleep, time

import jax
import numpy as np
from dsps.data_loaders import load_ssp_templates
from jax import random as jran
from mpi4py import MPI

from diffsky import phot_utils
from diffsky.data_loaders import mpi_utils
from diffsky.data_loaders.hacc_utils import lc_mock_production as lcmp
from diffsky.data_loaders.hacc_utils import lightcone_utils as hlu
from diffsky.data_loaders.hacc_utils import load_lc_cf
from diffsky.data_loaders.hacc_utils import load_lc_cf_synthetic as llcs
from diffsky.data_loaders.hacc_utils import metadata_sfh_mock
from diffsky.data_loaders.mock_utils import get_mock_version_name
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.experimental.sfh_model_calibrations import (
    load_diffsky_sfh_model_calibrations as ldup,
)
from diffsky.param_utils import diffsky_param_wrapper as dpw

DRN_LJ_CF_LCRC = "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest"
DRN_LJ_CF_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"

DRN_LJ_LC_LCRC = (
    "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/core-lc-6/output"
)
DRN_LJ_LC_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc_cores"

DRN_LJ_CROSSX_OUT_LCRC = "/lcrc/project/cosmo_ai/ahearin/LastJourney/lc-cf-diffsky"
DRN_LJ_CROSSX_OUT_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc-cf-diffsky"


SIM_NAME = "LastJourney"
DIFFSTARPOP_CALIBRATIONS = [
    "smdpl_dr1",
    "tng",
    "galacticus_in_plus_ex_situ",
]

ROMAN_HLTDS_PATCHES = [157, 158, 118, 119]


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "machine", help="Machine name where script is run", choices=["lcrc", "poboy"]
    )
    parser.add_argument("z_min", help="Minimum redshift", type=float)
    parser.add_argument("z_max", help="Maximum redshift", type=float)
    parser.add_argument("istart", help="First sky patch", type=int)
    parser.add_argument("iend", help="Last sky patch", type=int)

    parser.add_argument("drn_out", help="Output directory")
    parser.add_argument("mock_nickname", help="Nickname of the mock")

    parser.add_argument(
        "-batch_size", help="Size of photometry batches", type=int, default=20_000
    )

    parser.add_argument(
        "-roman_hltds",
        help="Use all patches overlapping with Roman HLTDS. Overrides istart and iend",
        default=0,
        choices=[0, 1],
        type=int,
    )

    parser.add_argument(
        "--ddf", action="store_true", help="Include LSST DDF sky patches"
    )

    parser.add_argument(
        "-fn_u_params",
        help="Best-fit diffsky parameters. Set to `sfh_model` to use a few specific calibrations",
        default="",
    )
    parser.add_argument(
        "-sfh_model",
        help="Assumed SFH model in diffsky calibration",
        default="tng",
        choices=DIFFSTARPOP_CALIBRATIONS,
    )

    parser.add_argument(
        "-indir_lc_data",
        help="Input drn storing lc_cores-*.*.hdf5",
        default=DRN_LJ_LC_LCRC,
    )
    parser.add_argument("-itest", help="Short test run?", type=int, default=0)
    parser.add_argument("-sim_name", help="Simulation name", default=SIM_NAME)
    parser.add_argument(
        "-synthetic_cores",
        help="Use synthetic cores instead of simulated cores",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-lgmp_min", help="Low-mass cutoff for synthetic cores", type=float, default=-1
    )
    parser.add_argument(
        "-lgmp_max", help="High-mass cutoff for synthetic cores", type=float, default=-1
    )

    args = parser.parse_args()
    machine = args.machine
    z_min = args.z_min
    z_max = args.z_max
    istart = args.istart
    iend = args.iend
    sfh_model = args.sfh_model
    drn_out = args.drn_out
    mock_nickname = args.mock_nickname

    roman_hltds = args.roman_hltds
    fn_u_params = args.fn_u_params
    itest = args.itest
    sim_name = args.sim_name
    synthetic_cores = args.synthetic_cores
    lgmp_min = args.lgmp_min
    lgmp_max = args.lgmp_max
    batch_size = args.batch_size

    mock_version_name = get_mock_version_name(mock_nickname)

    if synthetic_cores == 1:
        drn_out = os.path.join(drn_out, "synthetic_cores")

        try:
            assert lgmp_min != -1
            assert lgmp_max != -1
        except AssertionError:
            msg = f"When argument synthetic_cores={synthetic_cores} "
            msg += "must specify lgmp_min and lgmp_max"
            raise ValueError(msg)

    if fn_u_params == "sfh_model":
        drn_out = os.path.join(drn_out, sfh_model)
    os.makedirs(drn_out, exist_ok=True)

    if machine == "poboy":
        indir_lc_diffsky = DRN_LJ_CROSSX_OUT_POBOY
        indir_lc_data = DRN_LJ_CROSSX_OUT_POBOY
    elif machine == "lcrc":
        indir_lc_diffsky = DRN_LJ_CROSSX_OUT_LCRC
        indir_lc_data = DRN_LJ_LC_LCRC

    ran_key = jran.key(0)

    sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim(sim_name)

    if itest == 1:
        lc_patch_list = [0, 1]
    elif roman_hltds == 1:
        lc_patch_list = np.array(ROMAN_HLTDS_PATCHES).astype(int)
        if rank == 0:
            print("Making all lightcone patches for Roman HLTDS")
    elif args.ddf:
        fn_lc_decomp = os.path.join(indir_lc_data, "lc_cores-decomposition.txt")
        lc_patch_dict = hlu.get_lsst_ddf_patches(fn_lc_decomp)
        lc_patch_list = np.unique(
            np.concatenate([arr for arr in lc_patch_dict.values()])
        )
    else:
        lc_patch_list = np.arange(istart, iend).astype(int)

    output_timesteps = hlu.get_timesteps_in_zrange(sim_name, z_min, z_max)

    ssp_data = load_ssp_templates()

    #  Load diffsky model parameters
    if fn_u_params == "":
        param_collection = dpw.DEFAULT_PARAM_COLLECTION
        if rank == 0:
            print(
                "No input params detected. "
                "Using default diffsky model parameters DEFAULT_PARAM_COLLECTION"
            )
    elif fn_u_params == "sfh_model":
        u_param_arr = ldup.load_diffsky_u_params_for_sfh_model(sfh_model)
        u_param_collection = dpw.get_u_param_collection_from_u_param_array(u_param_arr)
        param_collection = dpw.get_param_collection_from_u_param_collection(
            *u_param_collection
        )
        if True:  # always do this for now
            if rank == 0:
                print("Ignoring pre-computed fit of SPS parameters")
            diffstarpop_sfh_model = param_collection[0]
            param_collection = (
                diffstarpop_sfh_model,
                *dpw.DEFAULT_PARAM_COLLECTION[1:],
            )

    else:
        if rank == 0:
            print(
                f"Reading diffsky parameter array from disk. Filename = {fn_u_params}"
            )
        u_param_arr = np.loadtxt(fn_u_params)
        u_param_collection = dpw.get_u_param_collection_from_u_param_array(u_param_arr)
        param_collection = dpw.get_param_collection_from_u_param_collection(
            *u_param_collection
        )

    n_z_phot_table = 15

    filter_nicknames = [f"lsst_{x}" for x in ("u", "g", "r", "i", "z", "y")]
    tcurves = lcmp.get_dsps_transmission_curves(filter_nicknames)

    # Get complete list of files to process
    fn_lc_list = []
    for lc_patch in lc_patch_list:
        for stepnum in output_timesteps:
            bn_lc_diffsky = lcmp.LC_CF_BNPAT.format(stepnum, lc_patch)
            fn_lc_diffsky = os.path.join(indir_lc_diffsky, bn_lc_diffsky)
            fn_lc_list.append(fn_lc_diffsky)

    if synthetic_cores == 0:
        fn_sizes = [os.path.getsize(fn) for fn in fn_lc_list]
    else:
        fn_sizes = []
        for fn in fn_lc_list:
            bn = os.path.basename(fn)
            stepnum, lc_patch = [int(x) for x in bn.split("-")[1].split(".")[:2]]
            fn_size = hlu._estimate_nhalos_sky_patch(sim_name, stepnum)
            fn_sizes.append(fn_size)
    rank_assignments, __ = mpi_utils.distribute_files_by_size(fn_sizes, nranks)
    fn_lc_list_for_rank = [fn_lc_list[i] for i in rank_assignments[rank]]

    print(f"\nFor rank = {rank}:")
    print(fn_lc_list_for_rank)
    print("\n")

    start_script = time()
    for fn_lc_diffsky in fn_lc_list_for_rank:
        gc.collect()

        bn_lc_diffsky = os.path.basename(fn_lc_diffsky)
        stepnum, lc_patch = [int(x) for x in bn_lc_diffsky.split("-")[1].split(".")[:2]]

        if rank == 0:
            print(f"...working on {os.path.basename(fn_lc_diffsky)}")

        ran_key, patch_key = jran.split(ran_key, 2)

        if synthetic_cores == 0:
            lc_data, diffsky_data = load_lc_cf.load_lc_diffsky_patch_data(
                fn_lc_diffsky, indir_lc_data
            )
        else:
            bn_in = os.path.basename(fn_lc_diffsky)
            bn_lc = os.path.basename(bn_in).replace(".diffsky_data.hdf5", ".hdf5")
            fn_lc_cores = os.path.join(indir_lc_data, bn_lc)
            patch_key, synthetic_lc_key = jran.split(patch_key, 2)
            lc_data, diffsky_data = llcs.load_lc_diffsky_patch_data(
                fn_lc_cores, sim_name, synthetic_lc_key, lgmp_min, lgmp_max
            )

        n_gals = len(lc_data["core_tag"])
        lc_data["stepnum"] = np.zeros(n_gals).astype(int) + stepnum
        lc_data["lc_patch"] = np.zeros(n_gals).astype(int) + lc_patch

        # Define redshift table used for magnitude interpolation
        _EPS = 1e-3
        z_max = lc_data["redshift_true"].max() + _EPS

        z_min_shell = lc_data["redshift_true"].min()
        z_min = z_min_shell - _EPS
        z_min_cutoff = 1e-3
        if z_min < z_min_cutoff:
            z_min = z_min_shell / 2
        z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

        # Precompute photometry at each element of the redshift table
        wave_eff_table = phot_utils.get_wave_eff_table(z_phot_table, tcurves)

        precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
            tcurves, ssp_data, z_phot_table, sim_info.cosmo_params
        )

        n_gals_orig = len(lc_data["core_tag"])

        n_batches = n_gals // batch_size
        print(f"{n_gals} total galaxies")
        print(f"Batch size = {batch_size:_}")
        print(f"Looping over {n_batches} batches of data\n")
        # Loop over batches of data
        phot_batches = []
        for istart in range(0, n_gals, batch_size):
            iend = min(istart + batch_size, n_gals)
            ran_key, batch_key = jran.split(ran_key)

            lc_data_batch = dict()
            for key in lc_data.keys():
                lc_data_batch[key] = lc_data[key][istart:iend]

            diffsky_data_batch = dict()
            for key in diffsky_data.keys():
                diffsky_data_batch[key] = diffsky_data[key][istart:iend]

            args = (
                sim_info,
                lc_data_batch,
                diffsky_data_batch,
                ssp_data,
                param_collection,
                precomputed_ssp_mag_table,
                z_phot_table,
                wave_eff_table,
                batch_key,
            )
            _res = lcmp.add_dbk_sed_quantities_to_mock(*args)
            phot_info_batch, lc_data_batch, diffsky_data_batch = _res
            phot_batches.append(_res)

        _cats = lcmp.concatenate_batched_phot_data(phot_batches)
        phot_info, lc_data, diffsky_data = _cats

        n_gals_check = len(lc_data["core_tag"])
        assert n_gals_orig == n_gals_check, "mismatch between orig and new lengths"
        print(f"Rank {rank}: Validating {n_gals_check} galaxies after batching")

        # Check every array has correct length
        for key, val in {**lc_data, **diffsky_data, **phot_info}.items():
            if val.shape[0] != n_gals_check:
                raise ValueError(
                    f"Array length mismatch: {key} has shape {val.shape}, "
                    f"expected first dim = {n_gals_check}"
                )

        gc.collect()
        jax.clear_caches()

        patch_key, morph_key = jran.split(patch_key, 2)
        diffsky_data = lcmp.add_morphology_quantities_to_diffsky_data(
            sim_info, phot_info, lc_data, diffsky_data, morph_key
        )

        diffsky_data = lcmp.add_black_hole_quantities_to_diffsky_data(
            lc_data, diffsky_data, phot_info
        )

        patch_key, nfw_key = jran.split(patch_key, 2)
        lc_data, diffsky_data = lcmp.reposition_satellites(
            sim_info, lc_data, diffsky_data, nfw_key
        )

        bn_out = lcmp.LC_MOCK_BNPAT.format(stepnum, lc_patch)
        fn_out = os.path.join(drn_out, bn_out)
        lcmp.write_lc_dbk_sed_mock_to_disk(
            fn_out, phot_info, lc_data, diffsky_data, filter_nicknames
        )
        metadata_sfh_mock.append_metadata(fn_out, sim_name, mock_version_name)

        bn_ssp_data = f"diffsky_{mock_version_name}_ssp_data.hdf5"
        fn_out_ssp_data = os.path.join(drn_out, bn_ssp_data)
        lcmp.write_lc_ssp_data_to_disk(drn_out, mock_version_name, tcurves, ssp_data)

        if rank == 0:
            print("All ranks completing file operations...", flush=True)

        gc.collect()
        jax.clear_caches()

    end_script = time()
    runtime = (end_script - start_script) / 60.0

    end_script = time()
    n_patches = len(lc_patch_list)
    runtime = (end_script - start_script) / 60.0
    if rank == 0:
        print("All ranks completing script...", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sleep(1)
    comm.Barrier()
    msg = f"Total runtime for {n_patches} patches = {runtime:.1f} minutes"
    if rank == 0:
        print(msg)
