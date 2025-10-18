"""Script to run sanity checks on cross-matched Last Journey lightcone data"""

import argparse
import os
from glob import glob

from diffsky.data_loaders.hacc_utils.lightcone_utils import (
    check_lc_cores_diffsky_data,
    write_lc_cores_diffsky_data_report_to_disk,
)

BN_GLOBPAT_LC_CORES = "lc_cores-*.*.diffsky_data.hdf5"
DRN_LJ_LC_LCRC = (
    "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/core-lc-6/output"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("drn_lc_cf", help="Directory with cross-matched diffsky_data")
    parser.add_argument(
        "-bnpat_lc_cf",
        help="Basename pattern for cross-matched diffsky_data",
        default="lc_cores-{0}.{1}.diffsky_data.hdf5",
    )

    parser.add_argument(
        "-drn_lc_cores", help="Directory with lc_cores", default=DRN_LJ_LC_LCRC
    )
    parser.add_argument(
        "-bnpat_lc_cores",
        help="Basename pattern for lc_cores",
        default=BN_GLOBPAT_LC_CORES,
    )

    args = parser.parse_args()
    drn_lc_cf = args.drn_lc_cf
    bnpat_lc_cf = args.bnpat_lc_cf
    drn_lc_cores = args.drn_lc_cores
    bnpat_lc_cores = args.bnpat_lc_cores

    fn_lc_cf_list = glob(os.path.join(drn_lc_cf, bnpat_lc_cf))
    fn_lc_cores_list = glob(os.path.join(drn_lc_cores, bnpat_lc_cores))

    bn_lc_cf_list = [os.path.basename(fn) for fn in fn_lc_cf_list]
    bn_lc_cores_list = [os.path.basename(fn) for fn in fn_lc_cores_list]

    # Check for missing files

    # all_good = True
    # failure_collector = []
    # for fn in fn_list:
    #     report = check_lc_cores_diffsky_data(fn)
    #     fn_report = fn.replace(".hdf5", ".report.txt")
    #     all_good_fn = write_lc_cores_diffsky_data_report_to_disk(report, fn_report)
    #     all_good = all_good & all_good_fn
    #     if not all_good_fn:
    #         bname = os.path.basename(fn)
    #         print(f"{bname} fails readiness test")
    #         failure_collector.append(bname)

    # if all_good:
    #     print("All lightcone patches pass all tests")
    # else:
    #     print("Some failures in the following lightcone patches:\n")
    #     for failing_bn in failure_collector:
    #         print(f"{failing_bn}")
