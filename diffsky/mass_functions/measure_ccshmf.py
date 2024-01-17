"""Module provides convenience functions used to measure the CCSHMF in SMDPL"""
import numpy as np

NHOST_BIN_MIN = 4
SMDPL_LGMP = np.log10(9.8e7)
NPTCL_CUT = 400


def get_lgmu_cutoff(lgmhost, lgmp_sim, nptcl_cut):
    lgmp_cutoff = lgmp_sim + np.log10(nptcl_cut)
    lgmu_cutoff = lgmp_cutoff - lgmhost
    return lgmu_cutoff


def smdpl_target_data_gen(
    logmhost_arr, host_lgmp, is_sat, lgmu_peak, nhost_min=NHOST_BIN_MIN
):
    for logmhost_bin_center in logmhost_arr:
        try:
            target_data = get_smdpl_target_data(
                logmhost_bin_center, host_lgmp, is_sat, lgmu_peak, nhost_min=nhost_min
            )
            logmhost_sample, lgmu_bins, lgcounts_target = target_data
            yield logmhost_sample, lgmu_bins, lgcounts_target
        except AssertionError:
            pass


def get_smdpl_target_data(
    logmhost_bin_center,
    host_lgmp,
    is_sat,
    lgmu_peak,
    lgmp_sim=SMDPL_LGMP,
    dlgmh_bin=0.25,
    lgmu_max=-0.2,
    n_lgmu_bins=50,
    nptcl_cut=NPTCL_CUT,
    nhost_min=NHOST_BIN_MIN,
):
    mmsk = np.abs(host_lgmp - logmhost_bin_center) < dlgmh_bin
    msk_hosts = mmsk & ~is_sat
    nhosts = np.sum(msk_hosts)

    msg = "Less than {0} hosts in this mass range".format(nhost_min)
    assert nhosts > nhost_min, msg

    logmhost_sample = np.median(host_lgmp[msk_hosts])

    lgmu_lo = get_lgmu_cutoff(logmhost_sample, lgmp_sim, nptcl_cut)
    lgmu_bins = np.linspace(lgmu_lo, lgmu_max, n_lgmu_bins)

    cuml_collector = []
    for lgmu in lgmu_bins:
        msk = mmsk & is_sat & (lgmu_peak > lgmu)
        cuml_collector.append(msk.sum())
    mean_cuml_sub_counts = np.array(cuml_collector) / nhosts

    msk_nonzero = mean_cuml_sub_counts > 0
    lgmu_bins = lgmu_bins[msk_nonzero]
    lgcounts_target = np.log10(mean_cuml_sub_counts[msk_nonzero])

    return logmhost_sample, lgmu_bins, lgcounts_target
