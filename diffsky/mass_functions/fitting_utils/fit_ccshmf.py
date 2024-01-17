"""ccshmf_fitter is a convenience function used to calibrate
ccshmf.predict_ccshmf, the CCSHMF of a host halo population

"""
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad

from ..ccshmf_model import DEFAULT_CCSHMF_PARAMS as P_INIT
from ..ccshmf_model import CCSHMF_Params, predict_ccshmf
from .fitting_helpers import jax_adam_wrapper


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff**2)


@jjit
def _loss_func_single_mhost(params, loss_data):
    target_lgmhost, target_lgmu_bins, target_lg_ccshmf = loss_data

    pred_lg_ccshmf = predict_ccshmf(params, target_lgmhost, target_lgmu_bins)
    loss_lg_ccshmf = _mse(pred_lg_ccshmf, target_lg_ccshmf)

    loss = loss_lg_ccshmf
    return loss


@jjit
def _loss_func_multi_mhost(params, loss_data):
    loss = 0.0
    for single_mhost_data in loss_data:
        loss = loss + _loss_func_single_mhost(params, single_mhost_data)
    return loss


_loss_and_grad_func = value_and_grad(_loss_func_multi_mhost, argnums=0)


def ccshmf_fitter(loss_data, n_steps=200, step_size=0.01, n_warmup=1):
    _res = jax_adam_wrapper(
        _loss_and_grad_func,
        P_INIT,
        loss_data,
        n_steps,
        step_size=step_size,
        n_warmup=n_warmup,
    )
    p_best, loss, loss_hist, params_hist, fit_terminates = _res
    p_best = CCSHMF_Params(*p_best)
    return p_best, loss, loss_hist, params_hist, fit_terminates
