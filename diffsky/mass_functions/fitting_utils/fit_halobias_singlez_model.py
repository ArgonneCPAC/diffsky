"""hmf_fitter is a convenience function used to calibrate
hmf.predict_hmf, the HMF of a host halo population as a function of mass and redshift

"""

from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad

from .. import halobias_singlez_model as hbm
from ..halobias_singlez_model import HALOBIAS_U_PARAMS as U_P_INIT
from .fitting_helpers import jax_adam_wrapper


@jjit
def _mae(pred, target):
    diff = pred - target
    return jnp.mean(jnp.abs(diff))


@jjit
def _loss_func_single_redshift(u_params, loss_data):
    params = hbm.get_bounded_halobias_params(u_params)
    target_lgmp, target_halobias = loss_data
    pred_halobias = hbm.predict_lgbias_kern(params, target_lgmp)
    loss = _mae(pred_halobias, target_halobias)
    return loss


_loss_and_grad_func = value_and_grad(_loss_func_single_redshift, argnums=0)


def halobias_singlez_fitter(
    loss_data, n_steps=200, step_size=0.05, n_warmup=1, u_p_init=U_P_INIT
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
