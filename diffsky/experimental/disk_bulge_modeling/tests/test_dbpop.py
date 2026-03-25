""""""

import numpy as np

from .. import dbpop


def test_frac_disk_dominated():
    logsm = np.linspace(6, 12, 100)
    fdd = dbpop.frac_disk_dominated(logsm)
    assert fdd.shape == logsm.shape
    assert np.all(fdd >= 0)
    assert np.all(fdd <= 1)
