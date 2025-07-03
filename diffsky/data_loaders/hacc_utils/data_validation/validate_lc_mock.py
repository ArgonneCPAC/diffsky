""" """

import os

import numpy as np

from .. import load_flat_hdf5

BNPAT_LC_MOCK = "lc_cores-{0}.{1}.diffsky_gals.hdf5"


def get_lc_mock_data_report(fn_lc_mock):
    report = dict()
    data = load_flat_hdf5(fn_lc_mock, dataset="data")

    msg = check_all_columns_are_finite(fn_lc_mock, data=data)

    if len(msg) > 0:
        report["finite_colums"] = msg

    return report


def write_lc_mock_report_to_disk(report, fn_lc_mock, drn_report):
    if len(report) > 0:
        bn_report = os.path.basename(fn_lc_mock).replace(".hdf5", ".report.txt")
        fn_report = os.path.join(drn_report, bn_report)
        with open(fn_report, "w") as fn_out:
            for line in report:
                fn_out.write(line + "\n")


def check_all_columns_are_finite(fn_lc_mock, data=None):
    bn = os.path.basename(fn_lc_mock)

    if data is None:
        data = load_flat_hdf5(fn_lc_mock, dataset="data")

    msg = []
    for key, arr in data.items():
        if not np.all(np.isfinite(arr)):
            s = f"Column {key} in {bn} has either NaN or inf"
            msg.append(s)

    return msg
