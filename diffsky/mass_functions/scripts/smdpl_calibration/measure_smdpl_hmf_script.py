"""Script to measure target data for SMDPL halo mass function fits"""

import argparse
import os
from glob import glob

import numpy as np

from diffsky.data_loaders.um_binary_loader import dtype as um_dtype
from diffsky.mass_functions import measure_hmf

DRN_OUT = "/home/ahearin/work/random/0711/SMDPL_HMF_TARGET_DATA"

DRN_LCRC = "/lcrc/project/halotools/UniverseMachine/SMDPL/sfr_catalogs_dr1_bestfit/"
BNPAT = "sfr_catalog_*.bin"
Z_TARGETS = np.arange(0.0, 5.0, 0.4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-drn", help="Directory of .bin files", default=DRN_LCRC)
    args = parser.parse_args()
    drn = args.drn

    fn_list = glob(os.path.join(drn, BNPAT))
    bn_list = [os.path.basename(fn) for fn in fn_list]
    a_arr = np.array([float(bn.split("_")[-1].replace(".bin", "")) for bn in bn_list])
    z_arr = 1 / a_arr - 1

    indx_z_list = [np.argmin(np.abs(z - z_arr)) for z in Z_TARGETS]

    os.makedirs(DRN_OUT, exist_ok=True)

    for indx_z in indx_z_list:
        fn = fn_list[indx_z]
        bn = os.path.basename(fn)
        print(f"...working on {bn}")

        data = np.fromfile(fn, dtype=um_dtype)
        # Mhalo is in units of Msun/h in sfr_catalog_*.bin, so we convert to Msun
        lgmp = np.log10(data["mp"] / measure_hmf.SMDPL_H)

        # All hosts and subs
        logmp_bins_subs, lgcuml_density_subs = measure_hmf.measure_smdpl_lg_cuml_hmf(
            lgmp
        )
        fn_out = os.path.join(DRN_OUT, bn.replace(".bin", ".subhalos.logmp_bins"))
        np.save(fn_out, logmp_bins_subs)

        fn_out = os.path.join(DRN_OUT, bn.replace(".bin", ".subhalos.lgcuml_density"))
        np.save(fn_out, lgcuml_density_subs)

        msk_cens = data["upid"] == -1
        logmp_bins_hosts, lgcuml_density_hosts = measure_hmf.measure_smdpl_lg_cuml_hmf(
            lgmp[msk_cens]
        )

        # Host halos only
        fn_out = os.path.join(DRN_OUT, bn.replace(".bin", ".hosthalos.logmp_bins"))
        np.save(fn_out, logmp_bins_hosts)

        fn_out = os.path.join(DRN_OUT, bn.replace(".bin", ".hosthalos.lgcuml_density"))
        np.save(fn_out, lgcuml_density_hosts)
