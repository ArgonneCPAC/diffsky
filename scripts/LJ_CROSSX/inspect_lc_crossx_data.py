"""Script to run sanity checks on cross-matched Last Journey lightcone data"""

import argparse
import os
from glob import glob

from diffsky.data_loaders.hacc_utils import hacc_core_utils as hcu
from diffsky.data_loaders.hacc_utils.lightcone_utils import (
    check_lc_cores_diffsky_data,
    write_lc_cores_diffsky_data_report_to_disk,
)

BN_GLOBPAT_LC_CORES = "lc_cores-{0}.{1}.hdf5"
BN_GLOBPAT_LC_CF = "lc_cores-{0}.{1}.diffsky_data.hdf5"
DRN_LJ_LC_LCRC = (
    "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/core-lc-6/output"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("drn_lc_cf", help="Directory with cross-matched diffsky_data")
    parser.add_argument(
        "z_min", help="Minimum redshift to check for matches", type=float
    )
    parser.add_argument(
        "z_max", help="Maximum redshift to check for matches", type=float
    )
    parser.add_argument(
        "patch_max", help="Maximum lc_patch to check for matches", type=int
    )

    parser.add_argument(
        "-bnpat_lc_cf",
        help="Basename pattern for cross-matched diffsky_data",
        default=BN_GLOBPAT_LC_CF.format("*", "*"),
    )

    parser.add_argument(
        "-drn_lc_cores", help="Directory with lc_cores", default=DRN_LJ_LC_LCRC
    )
    parser.add_argument(
        "-bnpat_lc_cores",
        help="Basename pattern for lc_cores",
        default=BN_GLOBPAT_LC_CORES.format("*", "*"),
    )

    args = parser.parse_args()
    drn_lc_cf = args.drn_lc_cf
    z_min = args.z_min
    z_max = args.z_max
    patch_max = args.patch_max

    bnpat_lc_cf = args.bnpat_lc_cf
    drn_lc_cores = args.drn_lc_cores
    bnpat_lc_cores = args.bnpat_lc_cores

    _res = hcu.get_timestep_range_from_z_range("LastJourney", z_min, z_max)
    timestep_min, timestep_max = _res[2:]

    fn_lc_cf_list = glob(os.path.join(drn_lc_cf, bnpat_lc_cf))
    fn_lc_cores_list = glob(os.path.join(drn_lc_cores, bnpat_lc_cores))

    bn_lc_cf_list = [os.path.basename(fn) for fn in fn_lc_cf_list]
    bn_lc_cores_list = [os.path.basename(fn) for fn in fn_lc_cores_list]

    missing_bn_collector = []
    for bn_lc_cores in bn_lc_cores_list:
        stepnum = int(bn_lc_cores.replace("lc_cores-", "").split(".")[0])
        lc_patch = int(bn_lc_cores.replace("lc_cores-", "").split(".")[1])

        if (
            (stepnum >= timestep_min)
            & (stepnum <= timestep_max)
            & (lc_patch <= patch_max)
        ):

            matching_bn_lc_cf = bn_lc_cores.replace(".hdf5", ".diffsky_data.hdf5")
            matching_fn_lc_cf = os.path.join(drn_lc_cf, matching_bn_lc_cf)

            if not os.path.isfile(matching_fn_lc_cf):
                missing_bn_collector.append(matching_bn_lc_cf)

    # Check for missing files

    # all_good = True
    # failure_collector = []
    # for fn in fn_list:
    #     report = check_lc_cores_diffsky_data(fn)
    #     fn_report = fn.replace(".hdf5", ".report.txt")
    #     all_good_fn = write_lc_cores_diffsky_data_report_to_disk(report, fn_report)
    #     all_good = all_good & all_good_fn
    #     if not all_good_fn:
    #         bname = os.path.basename(fn)
    #         print(f"{bname} fails readiness test")
    #         failure_collector.append(bname)

    # if all_good:
    #     print("All lightcone patches pass all tests")
    # else:
    #     print("Some failures in the following lightcone patches:\n")
    #     for failing_bn in failure_collector:
    #         print(f"{failing_bn}")
