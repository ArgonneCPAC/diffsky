"""
"""
import numpy as np

from ..hmf_kernels import DEFAULT_HMF_KERN_PARAMS, lg_hmf_kern


def test_lg_hmf_kern_evaluates():
    lgmu_arr = np.linspace(-6, 0, 500)
    res = lg_hmf_kern(DEFAULT_HMF_KERN_PARAMS, lgmu_arr)
    assert res.shape == lgmu_arr.shape
    assert np.all(np.isfinite(res))
