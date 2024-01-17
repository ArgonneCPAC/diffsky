"""
"""
import numpy as np
from jax import random as jran

from ...kernels.hmf_kernels import DEFAULT_HMF_KERN_PARAMS, lg_hmf_kern
from ..fit_hmf_kernels import _loss_func, _mse, hmf_kernel_fitter


def test_cshmf_kernel_fitter():
    ran_key = jran.PRNGKey(0)

    upran = jran.normal(ran_key, shape=(len(DEFAULT_HMF_KERN_PARAMS),)) * 0.1
    alt_params = np.array(DEFAULT_HMF_KERN_PARAMS) + upran

    lgmp_target = np.linspace(11, 15, 100)
    lg_cuml_hmf_target = lg_hmf_kern(alt_params, lgmp_target)

    res = hmf_kernel_fitter(lgmp_target, lg_cuml_hmf_target)
    p_best, loss_best = res[:2]

    loss_data = lgmp_target, lg_cuml_hmf_target
    loss_init = _loss_func(DEFAULT_HMF_KERN_PARAMS, loss_data)
    assert loss_init > loss_best

    pred_best = lg_hmf_kern(p_best, lgmp_target)
    recomputed_loss = _mse(pred_best, lg_cuml_hmf_target)
    assert np.allclose(recomputed_loss, loss_best, rtol=1e-3)
