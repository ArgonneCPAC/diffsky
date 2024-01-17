"""cshmf_kernel_fitter is a convenience function used to calibrate
ccshmf_kernels.lg_ccshmf_kern, the CCSHMF of an individual halo
"""
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad

from ..kernels.ccshmf_kernels import DEFAULT_CCSHMF_KERN_PARAMS as P_INIT
from ..kernels.ccshmf_kernels import CCSHMF_Params, lg_ccshmf_kern
from .fitting_helpers import jax_adam_wrapper


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff**2)


@jjit
def _loss_func(params, loss_data):
    lgmu_bins, target = loss_data
    pred = lg_ccshmf_kern(params, lgmu_bins)
    return _mse(pred, target)


_loss_and_grad_func = value_and_grad(_loss_func, argnums=0)


def cshmf_kernel_fitter(
    lgmu_bins, lgcounts_target, n_steps=200, step_size=0.01, n_warmup=1
):
    loss_data = lgmu_bins, lgcounts_target
    _res = jax_adam_wrapper(
        _loss_and_grad_func,
        P_INIT,
        loss_data,
        n_steps,
        step_size=step_size,
        n_warmup=n_warmup,
    )
    p_best, loss, loss_hist, __, fit_terminates = _res
    lgmu_plot = jnp.linspace(-5, 0.5, 500)
    p_best = CCSHMF_Params(*p_best)
    pred_best = lg_ccshmf_kern(p_best, lgmu_plot)
    return p_best, loss, loss_hist, lgmu_plot, pred_best, loss_data
