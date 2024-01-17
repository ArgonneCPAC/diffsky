"""
"""
import numpy as np
from jax import random as jran

from ...kernels.ccshmf_kernels import DEFAULT_CCSHMF_KERN_PARAMS, lg_ccshmf_kern
from ..fit_ccshmf_kernels import _loss_func, _mse, cshmf_kernel_fitter


def test_cshmf_kernel_fitter():
    ran_key = jran.PRNGKey(0)

    upran = jran.normal(ran_key, shape=(len(DEFAULT_CCSHMF_KERN_PARAMS),)) * 0.1
    alt_params = np.array(DEFAULT_CCSHMF_KERN_PARAMS) + upran

    lgmu_bins = np.linspace(-5, 0, 100)
    lgcounts_target = lg_ccshmf_kern(alt_params, lgmu_bins)

    res = cshmf_kernel_fitter(lgmu_bins, lgcounts_target)
    p_best, loss_best, __, __, __, loss_data = res

    loss_init = _loss_func(DEFAULT_CCSHMF_KERN_PARAMS, loss_data)
    assert loss_init > loss_best

    lgmu_target, target = loss_data[0:2]
    pred_best = lg_ccshmf_kern(p_best, lgmu_target)
    recomputed_loss = _mse(pred_best, target)
    assert np.allclose(recomputed_loss, loss_best, rtol=1e-3)
