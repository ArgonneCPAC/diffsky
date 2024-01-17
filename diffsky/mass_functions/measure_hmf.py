"""Module provides convenience functions used to measure the HMF in SMDPL"""
import numpy as np

SMDPL_VOL = 400.0**3
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
