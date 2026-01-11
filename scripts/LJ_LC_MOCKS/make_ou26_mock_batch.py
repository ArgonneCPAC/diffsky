"""Script to make an SED mock to populate a Last Journey lightcone

To run a unit test of this script:

python scripts/LJ_LC_MOCKS/make_ou26_mock_batch.py  poboy 0.08 0.1 0 1 ci_test_output ci_test_mock -sfh_model smdpl_dr1 -synthetic_cores 1 -lgmp_min 12.5 -lgmp_max 13.5
python scripts/LJ_LC_MOCKS/inspect_lc_mock.py ci_test_output/synthetic_cores/smdpl_dr1

python scripts/LJ_LC_MOCKS/make_ou26_mock_batch.py  poboy 0.08 0.1 0 1 ci_test_output ci_test_mock -cosmos_fit cosmos260105 -synthetic_cores 1 -lgmp_min 12.5 -lgmp_max 13.5
python scripts/LJ_LC_MOCKS/inspect_lc_mock.py ci_test_output/synthetic_cores/cosmos260105

python scripts/LJ_LC_MOCKS/make_ou26_mock_batch.py  poboy 0.08 0.1 0 1 ci_test_output ci_test_mock -cosmos_fit cosmos260105 -synthetic_cores 1 -lgmp_min 12.5 -lgmp_max 13.5 --no_dbk
python scripts/LJ_LC_MOCKS/inspect_lc_mock.py ci_test_output/synthetic_cores/cosmos260105 --no_dbk


"""  # noqa

import argparse
import gc
import os
import sys
from time import sleep, time

import h5py
import jax
import numpy as np
from dsps.data_loaders import load_ssp_templates
from jax import random as jran
from mpi4py import MPI

from diffsky import phot_utils
from diffsky.data_loaders import load_flat_hdf5, mpi_utils
from diffsky.data_loaders.hacc_utils import lc_mock as lcmp_repro
from diffsky.data_loaders.hacc_utils import lightcone_utils as hlu
from diffsky.data_loaders.hacc_utils import load_lc_cf
from diffsky.data_loaders.hacc_utils import load_lc_cf_synthetic as llcs
from diffsky.data_loaders.hacc_utils import metadata_mock
from diffsky.data_loaders.mock_utils import get_mock_version_name
from diffsky.experimental import mc_lightcone_halos as mclh
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.param_utils import get_mock_params as gmp

DRN_LJ_CF_LCRC = "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest"
DRN_LJ_CF_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"

DRN_LJ_LC_LCRC = (
    "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/core-lc-6/output"
)
DRN_LJ_LC_POBOY = "/Users/aphearin/work/DATA/LastJourney/core-lc-6"

DRN_LJ_CROSSX_OUT_LCRC = "/lcrc/project/cosmo_ai/ahearin/LastJourney/lc-cf-diffsky"
DRN_LJ_CROSSX_OUT_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc-cf-diffsky"


SIM_NAME = "LastJourney"

ROMAN_HLTDS_PATCHES = [157, 158, 118, 119]

LSST_FILTER_NICKNAMES = [f"lsst_{x}" for x in ("u", "g", "r", "i", "z", "y")]

ROMAN_FILTER_NICKNAMES = (
    "roman_F062",
    "roman_F087",
    "roman_F106",
    "roman_F129",
    "roman_F158",
    "roman_F184",
    "roman_F146",
    "roman_F213",
    "roman_Prism",
    "roman_Grism_1stOrder",
    "roman_Grism_0thOrder",
)
OUTPUT_FILTER_NICKNAMES = (*LSST_FILTER_NICKNAMES, *ROMAN_FILTER_NICKNAMES)

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
        "--roman_hltds",
        help="Use all patches overlapping with Roman HLTDS. Overrides istart and iend",
        action="store_true",
    )

    parser.add_argument(
        "--lsst_ddf",
        help="Use all patches overlapping with LSST DDF. Overrides istart and iend",
        action="store_true",
    )
    parser.add_argument(
        "--lsst_only", help="Use only LSST bandpasses", action="store_true"
    )

    parser.add_argument("-cosmos_fit", help="Best-fit diffsky parameters", default="")
    parser.add_argument(
        "-sfh_model", help="Assumed SFH model in diffsky calibration", default="tng"
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
    parser.add_argument(
        "--no_dbk",
        help="Exclude disk/bulge/knot SEDs in output mock",
        action="store_true",
    )
    parser.add_argument(
        "--no_sed",
        help="Exclude SEDs in output mock (use for SFH-only mocks)",
        action="store_true",
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
    lsst_ddf = args.lsst_ddf
    lsst_only = args.lsst_only
    cosmos_fit = args.cosmos_fit
    itest = args.itest
    sim_name = args.sim_name
    synthetic_cores = args.synthetic_cores
    lgmp_min = args.lgmp_min
    lgmp_max = args.lgmp_max
    batch_size = args.batch_size
    no_dbk = args.no_dbk
    no_sed = args.no_sed

    mock_version_name = get_mock_version_name(mock_nickname)

    if lsst_only:
        OUTPUT_FILTER_NICKNAMES = (*LSST_FILTER_NICKNAMES,)

    if synthetic_cores == 1:
        drn_out = os.path.join(drn_out, "synthetic_cores")

        try:
            assert lgmp_min != -1
            assert lgmp_max != -1
        except AssertionError:
            msg = f"When argument synthetic_cores={synthetic_cores} "
            msg += "must specify lgmp_min and lgmp_max"
            raise ValueError(msg)

    if cosmos_fit != "":
        subdrn = cosmos_fit
    elif sfh_model != "":
        subdrn = sfh_model
    else:
        subdrn = "default_model"
    drn_out = os.path.join(drn_out, subdrn)
    os.makedirs(drn_out, exist_ok=True)

    if machine == "poboy":
        indir_lc_diffsky = DRN_LJ_CROSSX_OUT_POBOY
        indir_lc_data = DRN_LJ_LC_POBOY
    elif machine == "lcrc":
        indir_lc_diffsky = DRN_LJ_CROSSX_OUT_LCRC
        indir_lc_data = DRN_LJ_LC_LCRC

    ran_key = jran.key(rank)

    sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim(sim_name)

    if itest == 1:
        lc_patch_list = [0, 1]
    else:
        lc_patch_list = []
        ignore_istart_iend = roman_hltds | lsst_ddf
        if ignore_istart_iend:
            if roman_hltds:
                lc_patch_list.extend(ROMAN_HLTDS_PATCHES)
                if rank == 0:
                    print("Making all lightcone patches for Roman HLTDS")
            if lsst_ddf:
                fn_lc_decomp = os.path.join(indir_lc_data, "lc_cores-decomposition.txt")
                lc_patch_dict = hlu.get_lsst_ddf_patches(fn_lc_decomp)
                lc_patch_list_lsst = np.unique(
                    np.concatenate([arr for arr in lc_patch_dict.values()])
                )
                lc_patch_list.extend(list(lc_patch_list_lsst))
                if rank == 0:
                    print("Making all lightcone patches for LSST DDF")
        else:
            lc_patch_list = np.arange(istart, iend).astype(int)
    lc_patch_list = np.array(lc_patch_list)
    if rank == 0:
        print(f"Making mock with lc_patch_list={lc_patch_list}")

    output_timesteps = hlu.get_timesteps_in_zrange(sim_name, z_min, z_max)

    ssp_data = load_ssp_templates()
    param_collection = gmp.get_param_collection_for_mock(
        cosmos_fit=cosmos_fit, sfh_model=sfh_model, rank=0
    )
    n_z_phot_table = 15

    tcurves = lcmp_repro.get_dsps_transmission_curves(OUTPUT_FILTER_NICKNAMES)
    assert len(tcurves) == len(OUTPUT_FILTER_NICKNAMES)

    # Get complete list of files to process
    fn_lc_list = []
    for lc_patch in lc_patch_list:
        for stepnum in output_timesteps:
            bn_lc_diffsky = lcmp_repro.LC_CF_BNPAT.format(stepnum, lc_patch)
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

        bn_in = os.path.basename(fn_lc_diffsky)
        bn_lc = os.path.basename(bn_in).replace(".diffsky_data.hdf5", ".hdf5")
        fn_lc_cores = os.path.join(indir_lc_data, bn_lc)
        lc_patch_info = llcs.get_lc_patch_info_from_lc_cores(fn_lc_cores, sim_name)

        bn_out = lcmp_repro.LC_MOCK_BNPAT.format(stepnum, lc_patch)
        fn_out = os.path.join(drn_out, bn_out)
        if os.path.exists(fn_out):
            os.remove(fn_out)

        if rank == 0:
            print(f"...working on {os.path.basename(fn_lc_diffsky)}")

        ran_key, patch_key = jran.split(ran_key, 2)

        if synthetic_cores == 0:
            with h5py.File(fn_lc_cores, "r") as hdf:
                nhalos_estimate = hdf["core_tag"].shape[0]
                z_min_shell = 1.0 / hdf["scale_factor"][:].max() - 1
                z_max_shell = 1.0 / hdf["scale_factor"][:].min() - 1
        elif synthetic_cores == 1:
            mean_nhalos = mclh.estimate_nhalos_in_lightcone(
                lgmp_min,
                lc_patch_info.z_lo,
                lc_patch_info.z_hi,
                lc_patch_info.sky_area_degsq,
            )
            nhalos_estimate = int(np.round(mean_nhalos))
            z_min_shell = lc_patch_info.z_lo
            z_max_shell = lc_patch_info.z_hi

        # Define redshift table used for magnitude interpolation
        _EPS = 1e-3
        z_min = min(lc_patch_info.z_lo, z_min_shell) - _EPS
        z_max = max(lc_patch_info.z_hi, z_max_shell) + _EPS

        z_min_cutoff = 1e-3
        if z_min < z_min_cutoff:
            z_min = z_min_shell / 2

        z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

        # Precompute photometry at each element of the redshift table
        wave_eff_table = phot_utils.get_wave_eff_table(z_phot_table, tcurves)

        precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
            tcurves, ssp_data, z_phot_table, sim_info.cosmo_params
        )

        print(f"Looping over {nhalos_estimate} halos with batch_size={batch_size}")
        for istart in range(0, nhalos_estimate, batch_size):
            iend = min(istart + batch_size, nhalos_estimate)

            patch_key, batch_key = jran.split(patch_key)

            if synthetic_cores == 0:
                lc_data_batch, diffsky_data_batch = (
                    load_lc_cf.load_lc_diffsky_patch_data(
                        fn_lc_diffsky, indir_lc_data, istart=istart, iend=iend
                    )
                )
            else:
                downsample_factor = nhalos_estimate / batch_size
                batch_key, synthetic_lc_key = jran.split(batch_key, 2)
                lc_data_batch, diffsky_data_batch = llcs.load_lc_diffsky_patch_data(
                    fn_lc_cores,
                    sim_name,
                    synthetic_lc_key,
                    lgmp_min,
                    lgmp_max,
                    downsample_factor=downsample_factor,
                )

            n_gals_batch = len(lc_data_batch["core_tag"])
            lc_data_batch["stepnum"] = np.zeros(n_gals_batch).astype(int) + stepnum
            lc_data_batch["lc_patch"] = np.zeros(n_gals_batch).astype(int) + lc_patch

            batch_key, vzero_key = jran.split(batch_key, 2)
            diffsky_data_batch = lcmp_repro.add_peculiar_velocity_to_mock(
                lc_data_batch, diffsky_data_batch, ran_key=vzero_key, impute_vzero=True
            )

            batch_key, phot_key = jran.split(batch_key, 2)
            if no_sed:
                raise NotImplementedError(
                    "SFH-only mock production not implemented yet"
                )
            else:
                args = (
                    sim_info,
                    lc_data_batch,
                    diffsky_data_batch,
                    ssp_data,
                    param_collection,
                    precomputed_ssp_mag_table,
                    z_phot_table,
                    wave_eff_table,
                    phot_key,
                )
                _res = lcmp_repro.add_dbk_phot_quantities_to_mock(*args)
                phot_info_batch, lc_data_batch, diffsky_data_batch = _res

                batch_key, morph_key = jran.split(batch_key, 2)
                diffsky_data_batch = (
                    lcmp_repro.add_morphology_quantities_to_diffsky_data(
                        sim_info,
                        phot_info_batch,
                        lc_data_batch,
                        diffsky_data_batch,
                        morph_key,
                    )
                )

                diffsky_data_batch = (
                    lcmp_repro.add_black_hole_quantities_to_diffsky_data(
                        lc_data_batch, diffsky_data_batch, phot_info_batch
                    )
                )

            if no_sed:
                raise NotImplementedError(
                    "SFH-only mock production not implemented yet"
                )
            else:
                if no_dbk:
                    lcmp_repro.write_batched_lc_sed_mock_to_disk(
                        fn_out,
                        phot_info_batch,
                        lc_data_batch,
                        diffsky_data_batch,
                        OUTPUT_FILTER_NICKNAMES,
                    )
                else:
                    lcmp_repro.write_batched_lc_dbk_sed_mock_to_disk(
                        fn_out,
                        phot_info_batch,
                        lc_data_batch,
                        diffsky_data_batch,
                        OUTPUT_FILTER_NICKNAMES,
                    )

            if synthetic_cores == 1:
                batch_key, nfw_key = jran.split(batch_key, 2)
                lc_data_batch, diffsky_data_batch = lcmp_repro.reposition_satellites(
                    sim_info, lc_data_batch, diffsky_data_batch, nfw_key
                )
                lcmp_repro.write_batched_mock_data(
                    fn_out,
                    lc_data_batch,
                    lcmp_repro.LC_DATA_NFW_KEYS_OUT,
                    dataset="data",
                )
                lcmp_repro.write_batched_mock_data(
                    fn_out,
                    diffsky_data_batch,
                    lcmp_repro.DIFFSKY_DATA_NFW_HOST_KEYS_OUT,
                    dataset="data",
                )

        if synthetic_cores == 0:
            lc_cores_poskeys = (
                "x",
                "y",
                "z",
                "top_host_idx",
                "redshift_true",
                "central",
                "logmp_obs",
            )
            lc_data_posinfo = load_flat_hdf5(
                fn_out, keys=lc_cores_poskeys, dataset="data"
            )
            diffsky_gals_posinfo = lc_data_posinfo

            patch_key, nfw_key = jran.split(patch_key, 2)
            lc_data_posinfo, diffsky_data_posinfo = lcmp_repro.reposition_satellites(
                sim_info, lc_data_posinfo, diffsky_gals_posinfo, nfw_key
            )

            lcmp_repro.write_batched_mock_data(
                fn_out, lc_data_posinfo, lcmp_repro.LC_DATA_NFW_KEYS_OUT, dataset="data"
            )
            lcmp_repro.write_batched_mock_data(
                fn_out,
                diffsky_data_posinfo,
                lcmp_repro.DIFFSKY_DATA_NFW_HOST_KEYS_OUT,
                dataset="data",
            )

        gc.collect()
        jax.clear_caches()

        if no_dbk:
            exclude_colnames = lcmp_repro.DBK_KEYS
        else:
            exclude_colnames = []

        metadata_mock.append_metadata(
            fn_out,
            sim_name,
            mock_version_name,
            z_phot_table,
            OUTPUT_FILTER_NICKNAMES,
            exclude_colnames=exclude_colnames,
            no_dbk=no_dbk,
        )

        if rank == 0:
            print("All ranks completing file operations...", flush=True)

            lcmp_repro.write_ancillary_data(
                drn_out,
                mock_version_name,
                sim_info,
                param_collection,
                tcurves,
                ssp_data,
            )

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
