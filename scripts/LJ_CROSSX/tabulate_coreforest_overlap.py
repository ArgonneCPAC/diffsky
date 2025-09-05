"""Script to tabulate lc_cores<-->coreforest file overlap for Last Journey"""

import argparse
import os
import pickle
from glob import glob
from time import time

import h5py
import numpy as np

from diffsky.data_loaders.hacc_utils import get_diffsky_info_from_hacc_sim
from diffsky.data_loaders.hacc_utils import lightcone_utils as hlu

DRN_LJ_LC_LCRC = (
    "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/core-lc-5/output"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-lc_cores_drn", help="Drn of lc_cores-*.*.hdf5", default=DRN_LJ_LC_LCRC
    )
    parser.add_argument("-outdrn", help="Output directory", default="LC_CF_XDATA")

    args = parser.parse_args()
    indir = args.lc_cores_drn
    outdrn = args.outdrn

    os.makedirs(outdrn, exist_ok=True)
    cf_xdict_outname = os.path.join(outdrn, "cf_xdict.pickle")
    lc_xdict_outname = os.path.join(outdrn, "lc_xdict.pickle")

    sim_info = get_diffsky_info_from_hacc_sim("LastJourney")

    lc_xdict = dict()

    cf_xdict = dict()
    for key in range(sim_info.num_subvols):
        cf_xdict[int(key)] = []

    lc_fnames = glob(os.path.join(indir, "lc_cores-*.hdf5"))

    n_lc_files = len(lc_fnames)
    n_out_check = int(0.1 * n_lc_files)
    n_batches = n_lc_files // n_out_check
    print(f"Starting loop over {n_lc_files} lightcone files\n")

    start = time()
    for counter, lc_fn in enumerate(lc_fnames):

        if counter % n_out_check == 0:
            n_batch_complete = counter // n_out_check
            elapsed_time = (time() - start) / 60.0
            print(f"{n_batch_complete}/{n_batches} done with file loop")
            print(f"Elapsed time = {elapsed_time:.1f} minutes\n")

        lc_bn = os.path.basename(lc_fn)
        stepnum, skypatch = hlu.get_stepnum_and_skypatch_from_lc_bname(lc_bn)

        with h5py.File(lc_fn, "r") as hdf:
            file_idx = hdf["coreforest_file_idx"][...]
        corefile_indices = np.unique(file_idx).astype(int)

        lc_xdict[(stepnum, skypatch)] = corefile_indices

        for corefile_idx in corefile_indices:
            cf_xdict[int(corefile_idx)].append((stepnum, skypatch))

    with open(cf_xdict_outname, "wb") as handle:
        pickle.dump(cf_xdict, handle)

    with open(lc_xdict_outname, "wb") as handle:
        pickle.dump(lc_xdict, handle)

    end = time()
    runtime = end - start
    print(f"Total runtime = {runtime:.1f} seconds")
    # Runtime = 4527.0 seconds (75.4 minutes)
