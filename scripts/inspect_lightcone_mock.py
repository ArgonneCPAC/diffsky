"""Script to run sanity checks on HACC lightcone mocks"""

import argparse
import os
from glob import glob

from diffsky.data_loaders.hacc_utils.lightcone_utils import (
    check_lc_cores_diffsky_data,
    write_lc_cores_diffsky_data_report_to_disk,
)

BN_GLOBPAT_LC_CORES = "lc_cores-*.*.diffsky_data.hdf5"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("drn", help="Directory storing lightcone mock")
    parser.add_argument("-bnpat", help="Basename pattern", default=BN_GLOBPAT_LC_CORES)

    args = parser.parse_args()
    drn = args.drn
    bnpat = args.bnpat
    fn_list = glob(os.path.join(drn, bnpat))

    all_good = True
    failure_collector = []
    for fn in fn_list:
        report = check_lc_cores_diffsky_data(fn)
        fn_report = fn.replace(".hdf5", ".report.txt")
        all_good_fn = write_lc_cores_diffsky_data_report_to_disk(report, fn_report)
        all_good = all_good & all_good_fn
        if not all_good_fn:
            bname = os.path.basename(fn)
            print(f"{bname} fails readiness test")
            failure_collector.append(bname)

    if all_good:
        print("All lightcone patches pass all tests")
    else:
        print("Some failures in the following lightcone patches:\n")
        for failing_bn in failure_collector:
            print(f"{failing_bn}")
