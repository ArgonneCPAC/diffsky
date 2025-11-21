""" """

from collections import namedtuple

import numpy as np
from jax import random as jran

from .. import io_utils as iou


def test_namedtuple_to_hdf5():
    ran_key = jran.key(0)
    KEYS = ("a", "b", "c", "d", "e", "f", "g")
    NDIM = len(KEYS)
    INDX = np.arange(NDIM).astype(int)

    n_tests = 10
    for __ in range(n_tests):
        ran_key, pname_key, pval_key = jran.split(ran_key, 3)
        indx = jran.choice(pname_key, INDX, shape=(NDIM,), replace=False)
        pnames = [KEYS[i] for i in indx]
        pvals = jran.uniform(pval_key, shape=(NDIM,))

        Params = namedtuple("Params", pnames)
        params = Params(*pvals)

        fn = "dummy.hdf5"
        iou.write_namedtuple_to_hdf5(params, fn)

        params2 = iou.load_namedtuple_from_hdf5(fn)

        assert np.allclose(params, params2)
