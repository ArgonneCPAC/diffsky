"""Script to make an SFH mock with DiffstarPop"""

import argparse
import os
from glob import glob
from time import time

from diffsky.data_loaders.hacc_utils import lc_mock_production, load_lc_cf
from jax import random as jran

DRN_LJ_CF_LCRC = "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest"
DRN_LJ_CF_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"

DRN_LJ_LC_LCRC = "/lcrc/project/cosmo_ai/mbuehlmann/LastJourney/core-lc-3/output"
DRN_LJ_LC_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc_cores"

DRN_LJ_CROSSX_OUT_LCRC = "/lcrc/project/halotools/LastJourney/lc-cf-diffsky"
DRN_LC_CF_XDATA_LCRC = os.path.join(DRN_LJ_CROSSX_OUT_LCRC, "LC_CF_XDATA")
DRN_LJ_CROSSX_OUT_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc-cf-diffsky"
DRN_LC_CF_XDATA_POBOY = os.path.join(DRN_LJ_CROSSX_OUT_POBOY, "LC_CF_XDATA")

LC_CF_BNPAT = "lc_cores-*.{}.diffsky_data.hdf5"
SIM_NAME = "LastJourney"
BNPAT_OUT = "diffsky_{0}.{1}.hdf5"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("indir_lc_cf", help="Output directory")
    parser.add_argument("lc_patch", help="Output directory")
    parser.add_argument("drn_out", help="Output directory")
    parser.add_argument("-bnpat_out", help="Basename pattern of output file")

    args = parser.parse_args()
    indir_lc_cf = args.indir_lc_cf
    lc_patch = args.lc_patch
    drn_out = args.drn_out

    ran_key = jran.key(0)

    sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim(SIM_NAME)

    fn_list = glob(os.path.join(indir_lc_cf, LC_CF_BNPAT.format(lc_patch)))
    print(f"Number of files = {len(fn_list)}")

    start = time()

    lc_data, diffsky_data = load_lc_cf.collect_lc_diffsky_data(
        fn_list, drn_lc_data=DRN_LJ_LC_LCRC
    )

    lc_data, diffsky_data = lc_mock_production.add_sfh_quantities_to_mock(
        sim_info, lc_data, diffsky_data, ran_key
    )

    bn_out = BNPAT_OUT.format(SIM_NAME, lc_patch)
    fn_out = os.path.join(drn_out, bn_out)
    lc_mock_production.write_lc_sfh_mock_to_disk(fn_out, lc_data, diffsky_data)

    end = time()
    runtime = (end - start) / 60.0
    print(f"Runtime to product lc_patch {lc_patch} = {runtime:.1f} minutes")
