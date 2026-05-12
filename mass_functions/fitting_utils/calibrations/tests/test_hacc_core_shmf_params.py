""" """

import numpy as np

from .. import hacc_core_shmf_params as hcshmf


def test_hmf_params():
    hmf_params = hcshmf.HMF_PARAMS
    for params in hmf_params:
        assert np.all(np.isfinite(params))
