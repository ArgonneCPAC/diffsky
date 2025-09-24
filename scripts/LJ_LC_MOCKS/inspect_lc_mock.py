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
    no_report_collector = []
    for fn_lc_mock in fn_lc_mock_list:
        bn_lc_mock = os.path.basename(fn_lc_mock)

        try:
            report = vlcm.get_lc_mock_data_report(fn_lc_mock)
            all_good = len(report) == 0
            if not all_good:
                vlcm.write_lc_mock_report_to_disk(report, fn_lc_mock, drn_report)
                print(f"{bn_lc_mock} fails readiness test")
                failure_collector.append(bn_lc_mock)
        except OSError:
            report = dict()
            report["report_does_not_exist"] = ["Unable to generate report"]
            print(f"Unable to generate report for {bn_lc_mock}")
            no_report_collector.append(bn_lc_mock)

    all_pass = (len(failure_collector) == 0) & (len(no_report_collector) == 0)
    if all_pass:
        print("Every lc_mock data file passes all tests")

    if len(failure_collector) > 0:
        print("\nSome failures in the following lightcone patches:\n")
        for failing_bn in failure_collector:
            print(f"{failing_bn}")

        fn_out = os.path.join(drn_report, "fn_list_fails_readiness.txt")
        with open(fn_out, "w") as fout:
            for failed_bn in failure_collector:
                fn_lc_mock = os.path.join(drn_mock, failed_bn)
                fout.write(fn_lc_mock + "\n")

    if len(no_report_collector) > 0:
        fn_out = os.path.join(drn_report, "fn_list_no_report.txt")
        with open(fn_out, "w") as fout:
            for no_report_bn in no_report_collector:
                fn_lc_mock = os.path.join(drn_mock, no_report_bn)
                fout.write(fn_lc_mock + "\n")
