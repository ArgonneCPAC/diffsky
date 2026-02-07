""""""

import numpy as np

from .. import disk_bulge_sizes as dbs
from .. import smzr_disk


def _mae(y1, y2):
    diff = y2 - y1
    return np.mean(np.abs(diff))


def test_lgr50_kern_bulge_default_model():
    n = 100
    marr = np.logspace(8.0, 11.5, n)
    lgmarr = np.log10(marr)
    ZZ = np.zeros(n)

    Z_TABLE = (0, 0.5, 1, 2)
    for redshift in Z_TABLE:
        lgr50_ek = np.log10(dbs._disk_median_r50(marr, ZZ + redshift))
        lgr50_sig_slope = smzr_disk._lgr50_kern_disk(
            lgmarr, ZZ + redshift, smzr_disk._DBS_DISK_SIZE_PARAMS
        )
        mae_diff = _mae(lgr50_ek, lgr50_sig_slope)
        assert mae_diff < 0.15
