"""
"""

from dsps.utils import _tw_cuml_kern
from jax import jit as jjit
from jax import lax, nn
from jax import numpy as jnp


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff * nn.sigmoid(k * (x - x0))


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - lax.log(lnarg) / k


@jjit
def smhm_loss_penalty(logsm_pred, logsm_target, penalty, dlgsm_max=0.5, h=0.1):
    """Penalty term for error in log10(Mstar)

    Parameters
    ----------
    logsm_pred : array, shape (n, )

    logsm_target : array, shape (n, )

    penalty : float
        Value to add to the loss for out-of-bounds predictions
        Should probably be ~1-10x the un-penalized loss

    logsm_pred : array, shape (n, )

    dlgsm_max : float, optional
        Parameter sets the maximum tolerated logsm difference
        Larger values produce a wider range in tolerated errors

    h : float, optional
        Transition speed of the triweight sigmoid used in the kernel

    Returns
    -------
    loss_penalty : array, shape (n, )

    """
    dlgsm = logsm_pred - logsm_target
    loss_penalty = _tophat_loss_kern(dlgsm, penalty, dlgsm_max, h)
    return loss_penalty


@jjit
def _tophat_loss_kern(x, penalty, dlgsm_max, h):
    w = 2 * (dlgsm_max - 6 * h)
    w = jnp.clip(w, 0.0)

    b = -w / 2.0
    x0 = b - 3 * h
    sigmoid_lo = penalty - _tw_cuml_kern(x, x0, h) * penalty

    b = w / 2.0
    x0 = b + 3 * h
    sigmoid_hi = _tw_cuml_kern(x, x0, h) * penalty

    loss = sigmoid_lo + sigmoid_hi
    return loss
