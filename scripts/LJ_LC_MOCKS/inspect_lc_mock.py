"""Script to run sanity checks on HACC lightcone mocks"""

import argparse
import gc
import os
from glob import glob
from time import time

import jax
import numpy as np
import yaml

from diffsky.data_loaders.hacc_utils.data_validation import validate_lc_mock as vlcm
from diffsky.data_loaders.mock_utils import get_mock_version_name

BN_GLOBPAT_LC_MOCK = "lc_cores-*.*.diffsky_gals*.hdf5"
BN_CHECKPAT_LC_MOCK = "lc_cores-{0}.{1}.diffsky_gals.hdf5"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config_yaml", help="YAML configuration file")
    parser.add_argument("-bnpat", help="Basename pattern", default=BN_GLOBPAT_LC_MOCK)
    parser.add_argument("-drn_report", help="Directory to write report", default="")

    parser.add_argument(
        "-drn_mock", help="Directory of mock overrides config_yaml", default=""
    )

    parser.add_argument(
        "-ignore_synth",
        help="Ignore synthetic halo files, default is False",
        action="store_true",
    )
    parser.add_argument(
        "-ignore_real",
        help="Ignore real halo files, default is False",
        action="store_true",
    )
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
        default=3,
        type=int,
    )
    parser.add_argument(
        "-infer_mockname",
        help="Infer mock_version_name from directory",
        action="store_true",
    )
    parser.add_argument(
        "-skip_slow_checks", help="Skip slow tests", action="store_true"
    )

    cl_args = parser.parse_args()
    config_yaml = cl_args.config_yaml
    bnpat = cl_args.bnpat
    drn_report = cl_args.drn_report
    no_dbk = cl_args.no_dbk
    no_sed = cl_args.no_sed
    n_files_to_check = cl_args.n_files_to_check

    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    mock_nickname = config["mock_nickname"]

    mock_version_name_in = config.get("mock_version_name", "")
    if cl_args.infer_mockname:
        fn_list = glob(os.path.join(config["drn_out"], mock_nickname + "_*"))
        drn_mock = fn_list[0]
        mock_version_name = os.path.basename(drn_mock)
    else:
        if mock_version_name_in == "":
            mock_version_name = get_mock_version_name(mock_nickname)
        else:
            mock_version_name = mock_version_name_in

    if cl_args.drn_mock == "":
        drn_mock = os.path.join(config["drn_out"], mock_version_name)
    else:
        drn_mock = cl_args.drn_mock

    fn_pat = os.path.join(drn_mock, bnpat)
    fn_list_all_mocks = glob(fn_pat)
    n_files_tot = len(fn_list_all_mocks)
    msg_no_mocks = f"No mocks detected with filename pattern {fn_pat}"
    assert n_files_tot > 1, msg_no_mocks

    missing_file_results = vlcm.check_for_missing_mock_patches(
        fn_list_all_mocks,
        BN_CHECKPAT_LC_MOCK,
        ignore_synth=cl_args.ignore_synth,
        ignore_real=cl_args.ignore_real,
    )

    if cl_args.ignore_real:
        print("\nNo missing mock files (ignoring)")
    else:
        if len(missing_file_results["missing_mock_files"]) > 0:
            print("The following mock files are missing:")
            for fn in missing_file_results["missing_mock_files"]:
                print(fn)
        else:
            print("\nNo missing mock files")

    if cl_args.ignore_synth:
        print("No missing synthetic mock files (ignoring)")
    else:
        if len(missing_file_results["missing_synth_files"]) > 0:
            print("The following synthetic files are missing:")
            for fn in missing_file_results["missing_synth_files"]:
                print(fn)
        else:
            print("No missing synthetic mock files")

    n_steps = len(missing_file_results["all_stepnums"])
    n_patches = len(missing_file_results["all_lc_patches"])
    n_files_to_check = min(n_files_to_check, n_files_tot)

    fn_list_mocks_to_test = []
    for __ in range(n_files_to_check):
        stepnum = np.random.choice(missing_file_results["all_stepnums"])
        lc_patch = np.random.choice(missing_file_results["all_lc_patches"])

        bn = BN_CHECKPAT_LC_MOCK.format(stepnum, lc_patch)
        fn = os.path.join(drn_mock, bn)
        if cl_args.ignore_real:
            pass
        else:
            fn_list_mocks_to_test.append(fn)

        fn_synth = os.path.join(drn_mock, bn.replace(".hdf5", ".synthetic_halos.hdf5"))
        if cl_args.ignore_synth:
            pass
        else:
            fn_list_mocks_to_test.append(fn_synth)

    fn_list_mocks_to_test = sorted(list(set(fn_list_mocks_to_test)))

    bn_list_mocks_to_test = [os.path.basename(fn) for fn in fn_list_mocks_to_test]
    print("\nTesting the following mocks:")
    for bn in bn_list_mocks_to_test:
        print("       " + bn)

    n_checked = 0
    start = time()
    all_good = True
    failure_collector = dict()
    no_report_collector = dict()
    for fn_lc_mock in fn_list_mocks_to_test:
        n_checked += 1
        jax.clear_caches()
        gc.collect()
        bn_lc_mock = os.path.basename(fn_lc_mock)

        try:
            report = vlcm.get_lc_mock_data_report(
                fn_lc_mock,
                no_dbk=no_dbk,
                no_sed=no_sed,
                skip_slow_checks=cl_args.skip_slow_checks,
            )
            all_good = len(report) == 0
            if not all_good:
                vlcm.write_lc_mock_report_to_disk(report, fn_lc_mock, drn_report)
                print(f"{bn_lc_mock} fails readiness test")
                failure_collector[bn_lc_mock] = report
        except OSError:
            report = dict()
            report["report_does_not_exist"] = ["Unable to generate report"]
            print(f"Unable to generate report for {bn_lc_mock}")
            no_report_collector[bn_lc_mock] = report

    all_pass = (len(failure_collector) == 0) & (len(no_report_collector) == 0)

    if len(failure_collector) > 0:
        print("\nSome failures in the following lightcone patches:\n")
        for failing_bn in failure_collector.keys():
            print(f"{failing_bn}")

        fn_out = os.path.join(drn_report, "fn_list_fails_readiness.txt")
        with open(fn_out, "w") as fout:
            for failed_bn in failure_collector.keys():
                fn_lc_mock = os.path.join(drn_mock, failed_bn)
                fout.write(fn_lc_mock + "\n")

    if len(no_report_collector) > 0:
        fn_out = os.path.join(drn_report, "fn_list_no_report.txt")
        with open(fn_out, "w") as fout:
            for no_report_bn in no_report_collector.keys():
                fn_lc_mock = os.path.join(drn_mock, no_report_bn)
                fout.write(fn_lc_mock + "\n")

    end = time()
    runtime = (end - start) / 60.0

    print(f"\nChecked {n_checked}/{n_files_tot} files in {runtime:.1f} minutes")
    if all_pass:
        print("\nEvery lc_mock data file passes all tests\n")
    else:
        if len(failure_collector) > 0:
            bn_lc_mock = list(failure_collector.keys())[0]
            fn_lc_mock = os.path.join(drn_mock, bn_lc_mock)
            report = failure_collector[bn_lc_mock]
            msg = f"{bn_lc_mock} fails readiness test:\n{report}"
            raise ValueError(msg)
        elif len(no_report_collector) > 0:
            bn_lc_mock = list(no_report_collector.keys())[0]
            msg = f"Unable to generate report for {bn_lc_mock}"
            raise ValueError(msg)
