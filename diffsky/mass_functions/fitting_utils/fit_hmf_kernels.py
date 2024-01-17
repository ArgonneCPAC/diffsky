"""hmf_kernel_fitter is a convenience function used to calibrate
hmf_kernels.lg_hmf_kern, the cumulative HMF at a single redshift

"""
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad

from ..kernels.hmf_kernels import DEFAULT_HMF_KERN_PARAMS as P_INIT
from ..kernels.hmf_kernels import HMF_Params, lg_hmf_kern
from .fitting_helpers import jax_adam_wrapper


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff**2)


@jjit
def _loss_func(params, loss_data):
    lgmp_bins, target = loss_data
    pred = lg_hmf_kern(params, lgmp_bins)
    return _mse(pred, target)


_loss_and_grad_func = value_and_grad(_loss_func, argnums=0)


def hmf_kernel_fitter(
    lgmp_bins, lgcounts_target, p_init=P_INIT, n_steps=200, step_size=0.01, n_warmup=1
):
    loss_data = lgmp_bins, lgcounts_target
    _res = jax_adam_wrapper(
        _loss_and_grad_func,
        p_init,
        loss_data,
        n_steps,
        step_size=step_size,
        n_warmup=n_warmup,
    )
    p_best, loss, loss_hist, __, fit_terminates = _res
    p_best = HMF_Params(*p_best)
    return p_best, loss, loss_hist
