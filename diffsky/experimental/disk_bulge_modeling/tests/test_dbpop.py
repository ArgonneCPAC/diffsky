""""""

import numpy as np
from jax import random as jran

from .. import dbpop


def test_frac_disk_dom_kern():
    ngals = 100
    logsm = np.linspace(6, 12, ngals)
    ZZ = np.zeros_like(logsm)

    for logssfr in (-12, -11, -10, -9, -8):
        fdd = dbpop._frac_disk_dom_kern(logsm, logssfr + ZZ)
        assert fdd.shape == logsm.shape
        assert np.all(fdd >= dbpop.FDD_MIN)
        assert np.all(fdd <= dbpop.FDD_MAX)
        assert np.all(np.diff(fdd) <= 0)
        assert np.any(np.diff(fdd) < 0)
