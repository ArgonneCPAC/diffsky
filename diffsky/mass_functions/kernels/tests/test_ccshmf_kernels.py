"""
"""
import numpy as np

from ..ccshmf_kernels import DEFAULT_CCSHMF_KERN_PARAMS, lg_ccshmf_kern


def test_lg_ccshmf_kern_evaluates():
    lgmu_arr = np.linspace(-6, 0, 500)
    res = lg_ccshmf_kern(DEFAULT_CCSHMF_KERN_PARAMS, lgmu_arr)
    assert res.shape == lgmu_arr.shape
    assert np.all(np.isfinite(res))
