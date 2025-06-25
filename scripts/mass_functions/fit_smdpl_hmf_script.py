"""Script to fit HMF target data for SMDPL.

This script uses data in the directory SMDPL_HISTOGRAMS created by
measure_smdpl_hmf_script.py
"""

import argparse

from diffsky.mass_functions.fitting_utils import fit_hmf_model
from diffsky.mass_functions.fitting_utils.diagnostics import hmf_fit_diagnostics
from diffsky.mass_functions.hmf_calibrations.smdpl_hmf_fitting_helpers import (
    get_loss_data,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("drn", help="Path to the SMDPL_HISTOGRAMS directory")
    parser.add_argument(
        "halotype", help="hosthalos or subhalos", choices=["hosthalos", "subhalos"]
    )
    args = parser.parse_args()
    drn = args.drn
    halotype = args.halotype

    loss_data_collector = get_loss_data(drn, halotype)

    _res = fit_hmf_model.hmf_fitter(loss_data_collector)
    p_best, loss, loss_hist, params_hist, fit_terminates = _res

    loss_best = fit_hmf_model._loss_func_multi_z(p_best, loss_data_collector)
    assert loss_best < 0.03, f"Poor HMF model fit: loss={loss_best:.4f}"

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
