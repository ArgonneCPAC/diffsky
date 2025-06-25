"""Script to fit HMF target data for SMDPL.

This script uses data in the directory SMDPL_HISTOGRAMS created by
measure_smdpl_hmf_script.py
"""

import argparse
import os
from glob import glob

import numpy as np

from diffsky.mass_functions.fitting_utils import fit_hmf_model
from diffsky.mass_functions.fitting_utils.diagnostics import hmf_fit_diagnostics

VOL_SMDPL = 400.0**3


def get_scale_from_bn(bn):
    a = float(".".join(bn.split("_")[2].split(".")[:2]))
    return a


def get_z_from_bn(bn):
    a = get_scale_from_bn(bn)
    return 1 / a - 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("drn", help="Path to the SMDPL_HISTOGRAMS directory")
    parser.add_argument(
        "halotype", help="hosthalos or subhalos", choices=["hosthalos", "subhalos"]
    )
    args = parser.parse_args()
    drn = args.drn
    halotype = args.halotype

    if halotype == "subhalos":
        fn_list = sorted(glob(os.path.join(drn, "*subhalos.lgcuml_density*.npy")))
        bn_list = [os.path.basename(fn) for fn in fn_list]
        fn_list_lgmp = [s.replace("lgcuml_density", "logmp_bins") for s in fn_list]
    elif halotype == "hosthalos":
        fn_list = sorted(glob(os.path.join(drn, "*hosthalos.lgcuml_density*.npy")))
        bn_list = [os.path.basename(fn) for fn in fn_list]
        fn_list_lgmp = [s.replace("lgcuml_density", "logmp_bins") for s in fn_list]
    else:
        raise ValueError(f"Unrecognized argument for halotype={halotype}")

    z_list = np.array([get_z_from_bn(bname) for bname in bn_list])

    loss_data_collector = []
    for iz, (fn, fn_lgmp) in enumerate(zip(fn_list, fn_list_lgmp)):
        lgcuml_density = np.load(fn)
        lgmp_bins = np.load(fn_lgmp)
        loss_data_iz = (z_list[iz], lgmp_bins, lgcuml_density)
        loss_data_collector.append(loss_data_iz)

    _res = fit_hmf_model.hmf_fitter(loss_data_collector)
    p_best, loss, loss_hist, params_hist, fit_terminates = _res

    hmf_fit_diagnostics.make_hmf_fit_plot(loss_data_collector, p_best)

    print("\n...Printing best-fit values...\n\n")

    for pname, pval in zip(p_best.ytp_params._fields, p_best.ytp_params):
        print(f"{pname}={pval:.3f},")
    print("\n")

    for pname, pval in zip(p_best.x0_params._fields, p_best.x0_params):
        print(f"{pname}={pval:.3f},")
    print("\n")

    for pname, pval in zip(p_best.lo_params._fields, p_best.lo_params):
        print(f"{pname}={pval:.3f},")
    print("\n")

    for pname, pval in zip(p_best.hi_params._fields, p_best.hi_params):
        print(f"{pname}={pval:.3f},")
    print("\n")
