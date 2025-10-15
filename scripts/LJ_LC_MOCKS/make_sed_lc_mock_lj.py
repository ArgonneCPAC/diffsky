"""Script to make an SED mock to populate a Last Journey lightcone"""

import argparse
import gc
import os
import pickle
from time import time

import numpy as np
from dsps.data_loaders import load_ssp_templates, load_transmission_curve
from jax import random as jran

from diffsky import phot_utils
from diffsky.data_loaders.hacc_utils import lc_mock_production as lcmp
from diffsky.data_loaders.hacc_utils import lightcone_utils as hlu
from diffsky.data_loaders.hacc_utils import load_lc_cf
from diffsky.data_loaders.hacc_utils import load_lc_cf_synthetic as llcs
from diffsky.data_loaders.hacc_utils import metadata_sfh_mock
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.param_utils import diffsky_param_wrapper as dpw

DRN_LJ_CF_LCRC = "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest"
DRN_LJ_CF_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"

DRN_LJ_LC_LCRC = (
    "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/core-lc-6/output"
)
DRN_LJ_LC_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc_cores"

DRN_LJ_CROSSX_OUT_LCRC = "/lcrc/project/halotools/LastJourney/lc-cf-diffsky"
DRN_LC_CF_XDATA_LCRC = os.path.join(DRN_LJ_CROSSX_OUT_LCRC, "LC_CF_XDATA")
DRN_LJ_CROSSX_OUT_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc-cf-diffsky"
DRN_LC_CF_XDATA_POBOY = os.path.join(DRN_LJ_CROSSX_OUT_POBOY, "LC_CF_XDATA")

LC_XDICT_BNAME = "lc_xdict.pickle"

SIM_NAME = "LastJourney"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "machine", help="Machine name where script is run", choices=["lcrc", "poboy"]
    )
    parser.add_argument("z_min", help="Minimum redshift", type=float)
    parser.add_argument("z_max", help="Maximum redshift", type=float)
    parser.add_argument("istart", help="First sky patch", type=int)
    parser.add_argument("iend", help="Last sky patch", type=int)

    parser.add_argument("drn_out", help="Output directory")
    parser.add_argument("-fn_u_params", help="Best-fit diffsky parameters", default="")

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
    drn_out = args.drn_out

    fn_u_params = args.fn_u_params
    itest = args.itest
    sim_name = args.sim_name
    synthetic_cores = args.synthetic_cores
    lgmp_min = args.lgmp_min
    lgmp_max = args.lgmp_max

    if synthetic_cores == 1:
        drn_out = os.path.join(drn_out, "synthetic_cores")

        try:
            assert lgmp_min != -1
            assert lgmp_max != -1
        except AssertionError:
            msg = f"When argument synthetic_cores={synthetic_cores} "
            msg += "must specify lgmp_min and lgmp_max"
            raise ValueError(msg)

    os.makedirs(drn_out, exist_ok=True)

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

    sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim(sim_name)

    if itest == 1:
        lc_patch_list = [0, 1]
    else:
        lc_patch_list = np.arange(istart, iend).astype(int)

    ssp_data = load_ssp_templates()
    if fn_u_params == "":
        param_collection = dpw.DEFAULT_PARAM_COLLECTION
    else:
        u_param_arr = np.loadtxt(fn_u_params)
        u_param_collection = dpw.get_u_param_collection_from_u_param_array(u_param_arr)
        param_collection = dpw.get_param_collection_from_u_param_collection(
            *u_param_collection
        )

    n_z_phot_table = 15
    z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)
    filter_nicknames = [f"lsst_{x}" for x in ("u", "g", "r", "i", "z", "y")]
    bn_pat_list = [name + "*" for name in filter_nicknames]
    tcurves = [load_transmission_curve(bn_pat=bn_pat) for bn_pat in bn_pat_list]
    wave_eff_table = phot_utils.get_wave_eff_table(z_phot_table, tcurves)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, sim_info.cosmo_params
    )

    start_script = time()
    for lc_patch in lc_patch_list:
        print(f"Working on lc_patch={lc_patch}")

        ran_key, patch_key, shuffle_key = jran.split(ran_key, 3)

        lc_patch_info_list = sorted(
            hlu.get_lc_patches_in_zrange(
                sim_name, lc_xdict, z_min, z_max, patch_list=[lc_patch]
            )
        )
        fn_list_lc_patch = [
            os.path.join(indir_lc_diffsky, lcmp.LC_CF_BNPAT.format(*patch_info))
            for patch_info in lc_patch_info_list
        ]
        bn_list_lc_patch = [os.path.basename(fn) for fn in fn_list_lc_patch]

        indx_all_steps = np.arange(len(lc_patch_info_list)).astype(int)

        print(f"lc_patch_info_list={lc_patch_info_list}")
        print(f"bn_list_lc_patch={bn_list_lc_patch}")

        start = time()
        for indx_step in indx_all_steps:
            fn_lc_diffsky = fn_list_lc_patch[indx_step]
            stepnum = lc_patch_info_list[indx_step][0]
            print(f"Working on={os.path.basename(fn_lc_diffsky)}")
            print(f"stepnum={stepnum}")

            if synthetic_cores == 0:
                lc_data, diffsky_data = load_lc_cf.load_lc_diffsky_patch_data(
                    fn_lc_diffsky, indir_lc_data
                )
            else:
                bn_in = os.path.basename(fn_lc_diffsky)
                bn_lc = os.path.basename(bn_in).replace(".diffsky_data.hdf5", ".hdf5")
                fn_lc_cores = os.path.join(indir_lc_data, bn_lc)
                lc_data, diffsky_data = llcs.load_lc_diffsky_patch_data(
                    fn_lc_cores, sim_name, ran_key, lgmp_min, lgmp_max
                )

            patch_key, sed_key = jran.split(patch_key, 2)
            args = (
                sim_info,
                lc_data,
                diffsky_data,
                ssp_data,
                param_collection,
                precomputed_ssp_mag_table,
                z_phot_table,
                wave_eff_table,
                sed_key,
            )
            phot_info, lc_data, diffsky_data = lcmp.add_sed_quantities_to_mock(*args)

            diffsky_data = lcmp.add_morphology_quantities_to_diffsky_data(
                phot_info, lc_data, diffsky_data
            )

            patch_key, nfw_key = jran.split(patch_key, 2)
            lc_data, diffsky_data = lcmp.reposition_satellites(
                sim_info, lc_data, diffsky_data, nfw_key
            )

            bn_out = lcmp.LC_MOCK_BNPAT.format(stepnum, lc_patch)
            fn_out = os.path.join(drn_out, bn_out)
            lcmp.write_lc_sed_mock_to_disk(
                fn_out, phot_info, lc_data, diffsky_data, filter_nicknames
            )
            metadata_sfh_mock.append_metadata(fn_out, sim_name)

            del lc_data
            del diffsky_data
            gc.collect()

        end = time()
        runtime = (end - start) / 60.0

    end_script = time()
    n_patches = len(lc_patch_list)
    runtime = (end_script - start_script) / 60.0
    msg = f"Total runtime for {n_patches} patches = {runtime:.1f} minutes"
    print(msg)
