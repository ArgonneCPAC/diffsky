"""Script to fit HMF target data for SMDPL.

This script uses data in the directory SMDPL_HISTOGRAMS created by
measure_smdpl_hmf_script.py
"""

import argparse
import os
from glob import glob

import numpy as np

from diffsky.mass_functions.fitting_utils import fit_hmf_model

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
    args = parser.parse_args()
    drn = args.drn

    fn_list_subs = sorted(glob(os.path.join(drn, "*subhalos.lgcuml_density*.npy")))
    bn_list_subs = [os.path.basename(fn) for fn in fn_list_subs]
    fn_list_hosts = sorted(glob(os.path.join(drn, "*hosthalos.lgcuml_density*.npy")))
    bn_list_hosts = [os.path.basename(fn) for fn in fn_list_hosts]

    fn_list_subs_lgmp = [
        s.replace("lgcuml_density", "logmp_bins") for s in fn_list_subs
    ]
    fn_list_hosts_lgmp = [
        s.replace("lgcuml_density", "logmp_bins") for s in fn_list_hosts
    ]

    z_list = np.array([get_z_from_bn(bname) for bname in bn_list_hosts])

    lgmp_bins = np.load(os.path.join(drn, "lgmp_bins.npy"))

    loss_data_collector = []
    for iz, (fn, fn_lgmp) in enumerate(zip(fn_list_hosts, fn_list_hosts_lgmp)):
        lgcuml_density = np.load(fn)
        lgmp_bins = np.load(fn_lgmp)
        loss_data_iz = (z_list[iz], lgmp_bins, lgcuml_density)
        loss_data_collector.append(loss_data_iz)

    _res = fit_hmf_model.hmf_fitter(loss_data_collector)
    p_best, loss, loss_hist, params_hist, fit_terminates = _res
