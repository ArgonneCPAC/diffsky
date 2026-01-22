"""Script to run sanity checks on HACC lightcone mocks"""

import argparse
import os
from glob import glob

import numpy as np

from diffsky.data_loaders.hacc_utils.data_validation import validate_lc_mock as vlcm

BN_GLOBPAT_LC_MOCK = "lc_cores-*.*.diffsky_gals.hdf5"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("drn_mock", help="Directory storing lightcone mock")
    parser.add_argument("-bnpat", help="Basename pattern", default=BN_GLOBPAT_LC_MOCK)
    parser.add_argument("-drn_report", help="Directory to write report", default="")
    parser.add_argument(
        "--no_dbk",
        help="disk/bulge/knot quantities are not in the mock",
        action="store_true",
    )
    parser.add_argument(
        "--no_sed",
        help="SEDs are not in the mock (SFH-only mocks)",
        action="store_true",
    )
    parser.add_argument(
        "-n_files_to_check",
        help="Number of randomly selected files to check",
        default=10,
        type=int,
    )

    args = parser.parse_args()
    drn_mock = args.drn_mock
    bnpat = args.bnpat
    drn_report = args.drn_report
    no_dbk = args.no_dbk
    no_sed = args.no_sed
    n_files_to_check = args.n_files_to_check

    fn_pat = os.path.join(drn_mock, bnpat)
    fn_list_all_mocks = glob(fn_pat)
    n_files_tot = len(fn_list_all_mocks)
    msg_no_mocks = f"No mocks detected with filename pattern {fn_pat}"
    assert n_files_tot > 1, msg_no_mocks

    fn_list_mocks_to_test = np.random.choice(
        fn_list_all_mocks, n_files_to_check, replace=False
    )
    bn_list_mocks_to_test = [os.path.basename(fn) for fn in fn_list_mocks_to_test]
    print("\n Testing the following mocks:")
    for bn in bn_list_mocks_to_test:
        print(bn + "\n")

    all_good = True
    failure_collector = []
    no_report_collector = []
    for fn_lc_mock in fn_list_mocks_to_test:
        bn_lc_mock = os.path.basename(fn_lc_mock)

        try:
            report = vlcm.get_lc_mock_data_report(
                fn_lc_mock,
                no_dbk=no_dbk,
                no_sed=no_sed,
            )
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

    print(f"Checked {n_files_to_check}/{n_files_tot} mock files")
    if all_pass:
        print("Every lc_mock data file passes all tests")
    else:
        raise ValueError("Some lightcone readiness tests fail")
