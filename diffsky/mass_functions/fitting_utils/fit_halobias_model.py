""" """

from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad

from .. import halobias_model as hbm
from .fitting_helpers import jax_adam_wrapper


@jjit
def _mae(pred, target):
    diff = pred - target
    return jnp.mean(jnp.abs(diff))


@jjit
def _loss_func_single_redshift(u_params, loss_data):
    params = hbm.get_bounded_halobias_params(u_params)
    target_lgmp, target_halobias, target_z = loss_data
    pred_halobias = hbm.predict_lgbias_kern(params, target_lgmp, target_z)
    loss = _mae(pred_halobias, target_halobias)
    return loss


@jjit
def _loss_func_multi_z(params, loss_data):
    loss = 0.0
    for single_z_data in loss_data:
        loss = loss + _loss_func_single_redshift(params, single_z_data)
    n_redshifts = len(loss_data)
    loss = loss / n_redshifts
    return loss


_loss_and_grad_func = value_and_grad(_loss_func_multi_z, argnums=0)


def halobias_model_fitter(
    loss_data, n_steps=200, step_size=0.05, n_warmup=1, u_p_init=hbm.HALOBIAS_U_PARAMS
):
    _res = jax_adam_wrapper(
        _loss_and_grad_func,
        u_p_init,
        loss_data,
        n_steps,
        step_size=step_size,
        n_warmup=n_warmup,
    )
    u_p_best, loss, loss_hist, params_hist, fit_terminates = _res
    p_best = hbm.get_bounded_halobias_params(u_p_best)
    return p_best, loss, loss_hist, params_hist, fit_terminates
