"""Script to run sanity checks on HACC lightcone mocks"""

import argparse
import os
from glob import glob

from diffsky.data_loaders.hacc_utils.data_validation import validate_lc_mock as vlcm

BN_GLOBPAT_LC_MOCK = "lc_cores-*.*.diffsky_gals.hdf5"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("drn_mock", help="Directory storing lightcone mock")
    parser.add_argument("-bnpat", help="Basename pattern", default=BN_GLOBPAT_LC_MOCK)
    parser.add_argument("-drn_report", help="Directory to write report", default="")

    args = parser.parse_args()
    drn_mock = args.drn_mock
    bnpat = args.bnpat
    drn_report = args.drn_report

    fn_lc_mock_list = glob(os.path.join(drn_mock, bnpat))

    all_good = True
    failure_collector = []
    for fn_lc_mock in fn_lc_mock_list:
        report = vlcm.get_lc_mock_data_report(fn_lc_mock)
        all_good = len(report) == 0
        if not all_good:
            vlcm.write_lc_mock_report_to_disk(report, fn_lc_mock, drn_report)

            bn_lc_mock = os.path.basename(fn_lc_mock)
            print(f"{bn_lc_mock} fails readiness test")
            failure_collector.append(bn_lc_mock)

    if len(failure_collector) == 0:
        print("Every lc_mock data file passes all tests")
    else:
        print("\nSome failures in the following lightcone patches:\n")
        for failing_bn in failure_collector:
            print(f"{failing_bn}")
