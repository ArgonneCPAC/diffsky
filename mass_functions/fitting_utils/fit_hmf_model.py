"""hmf_fitter is a convenience function used to calibrate
hmf.predict_hmf, the HMF of a host halo population as a function of mass and redshift

"""

from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad

from ..hmf_model import DEFAULT_HMF_PARAMS as P_INIT
from ..hmf_model import HMF_Params, predict_cuml_hmf
from .fitting_helpers import jax_adam_wrapper


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff**2)


@jjit
def _loss_func_single_redshift(params, loss_data):
    redshift, target_lgmp, target_hmf = loss_data
    pred_hmf = predict_cuml_hmf(params, target_lgmp, redshift)
    loss = _mse(pred_hmf, target_hmf)
    return loss


@jjit
def _loss_func_multi_z(params, loss_data):
    loss = 0.0
    for single_z_data in loss_data:
        loss = loss + _loss_func_single_redshift(params, single_z_data)
    return loss


_loss_and_grad_func = value_and_grad(_loss_func_multi_z, argnums=0)


def hmf_fitter(loss_data, n_steps=200, step_size=0.01, n_warmup=1, p_init=P_INIT):
    _res = jax_adam_wrapper(
        _loss_and_grad_func,
        p_init,
        loss_data,
        n_steps,
        step_size=step_size,
        n_warmup=n_warmup,
    )
    p_best, loss, loss_hist, params_hist, fit_terminates = _res
    p_best = HMF_Params(*p_best)
    return p_best, loss, loss_hist, params_hist, fit_terminates
