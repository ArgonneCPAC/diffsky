"""Module provides convenience functions used to measure the HMF in SMDPL"""

import numpy as np

SMDPL_H = 0.6777
SMDPL_VOL = (400.0 / SMDPL_H) ** 3
SMDPL_LGMP = np.log10(9.8e7)
NPTCL_CUT = 300
SMDPL_LGMP_CUT = np.log10(NPTCL_CUT) + SMDPL_LGMP
N_LGMP_BINS = 50
LGMP_BINS = np.linspace(SMDPL_LGMP_CUT, 16, N_LGMP_BINS)


def measure_smdpl_lg_cuml_hmf(logmp_data, logmp_bins=None, nhalos_min=10):
    if logmp_bins is None:
        logmp_bins_max = logmp_data.max()
        logmp_bins = np.linspace(SMDPL_LGMP_CUT, logmp_bins_max, N_LGMP_BINS)
    counts = np.array([np.sum(logmp_data >= logmp) for logmp in logmp_bins])
    cuml_density = counts / SMDPL_VOL

    msk_nonzero = counts >= nhalos_min
    logmp_bins = logmp_bins[msk_nonzero]
    lgcounts_target = np.log10(cuml_density[msk_nonzero])
    return logmp_bins, lgcounts_target


def measure_cuml_hmf_target_data_counts(logmp_data, logmp_bins):
    counts = [np.sum(logmp_data >= logmp) for logmp in logmp_bins]
    counts = np.array(counts).astype(int)
    return counts
