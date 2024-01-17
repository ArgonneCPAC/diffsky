"""Implementation of jax_adam_wrapper that is compatible with namedtuple parameters
"""
from copy import deepcopy

import numpy as np
from jax.example_libraries import optimizers as jax_opt


def jax_adam_wrapper(
    loss_and_grad_func,
    params_init,
    loss_data,
    n_step,
    n_warmup=0,
    step_size=0.01,
    warmup_n_step=50,
    warmup_step_size=None,
    tol=0.0,
):
    """Convenience function wrapping JAX's Adam optimizer used to
    minimize the loss function loss_func.

    Starting from params_init, we take n_step steps down the gradient
    to calculate the returned value params_step_n.

    Parameters
    ----------
    loss_and_grad_func : callable
        Differentiable function to minimize.

        Must accept inputs (params, data) and return a scalar and its gradients

    params_init : ndarray of shape (n_params, )
        Initial guess at the parameters

    loss_data : sequence
        Sequence of floats and arrays storing whatever data is needed
        to compute loss_func(params_init, loss_data)

    n_step : int
        Number of steps to walk down the gradient

    n_warmup : int, optional
        Number of warmup iterations. At the end of the warmup, the best-fit parameters
        are used as input parameters to the final burn. Default is zero.

    warmup_n_step : int, optional
        Number of Adam steps to take during warmup. Default is 50.

    warmup_step_size : float, optional
        Step size to use during warmup phase. Default is 5*step_size.

    step_size : float, optional
        Step size parameter in the Adam algorithm. Default is 0.01.

    Returns
    -------
    params_step_n : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps

    loss : float
        Final value of the loss

    loss_arr : ndarray of shape (n_step, )
        Stores the value of the loss at each step

    params_arr : ndarray of shape (n_step, n_params)
        Stores the value of the model params at each step

    fit_terminates : int
        0 if NaN or inf is encountered by the fitter, causing termination before n_step
        1 for a fit that terminates with no such problems

    """
    if warmup_step_size is None:
        warmup_step_size = 5 * step_size

    loss_init = float("inf")
    p_warmup = params_init
    for i in range(n_warmup):
        fit_results = _jax_adam_wrapper(
            loss_and_grad_func,
            p_warmup,
            loss_data,
            warmup_n_step,
            step_size=warmup_step_size,
            tol=tol,
        )
        p_warmup = fit_results[0]
        loss_init = fit_results[1]

    if loss_init > tol:
        fit_results = _jax_adam_wrapper(
            loss_and_grad_func,
            p_warmup,
            loss_data,
            n_step,
            step_size=step_size,
            tol=tol,
        )

    if len(fit_results[2]) < n_step:
        fit_terminates = 0
    else:
        fit_terminates = 1
    return (*fit_results, fit_terminates)


def _jax_adam_wrapper(
    loss_and_grad_func, params_init, loss_data, n_step, step_size=0.01, tol=0.0
):
    """Convenience function wrapping JAX's Adam optimizer used to
    minimize the loss function loss_func.

    Starting from params_init, we take n_step steps down the gradient
    to calculate the returned value params_step_n.

    Parameters
    ----------
    loss_and_grad_func : callable
        Differentiable function to minimize.

        Must accept inputs (params, data) and return a scalar loss and its gradients

    params_init : ndarray of shape (n_params, )
        Initial guess at the parameters

    loss_data : sequence
        Sequence of floats and arrays storing whatever data is needed
        to compute loss_func(params_init, loss_data)

    n_step : int
        Number of steps to walk down the gradient

    step_size : float, optional
        Step size parameter in the Adam algorithm. Default is 0.01

    Returns
    -------
    params_step_n : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps

    loss : float
        Final value of the loss

    loss_arr : ndarray of shape (n_step, )
        Stores the value of the loss at each step

    params_arr : ndarray of shape (n_step, n_params)
        Stores the value of the model params at each step

    """
    loss_collector = []
    params_collector = []

    best_fit_params = deepcopy(params_init)
    loss_best, __ = loss_and_grad_func(best_fit_params, loss_data)

    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(params_init)

    for istep in range(n_step):
        params_istep = get_params(opt_state)

        loss_istep, grads = loss_and_grad_func(params_istep, loss_data)
        loss_collector.append(loss_istep)
        params_collector.append(params_istep)

        if loss_istep < loss_best:
            loss_best = loss_istep
            best_fit_params = params_istep

        opt_state = opt_update(istep, grads, opt_state)

        if loss_best < tol:
            best_fit_params = best_fit_params
            loss_arr = np.array(loss_collector)
            return best_fit_params, loss_best, loss_arr, params_collector

    loss_arr = np.array(loss_collector)

    return best_fit_params, loss_best, loss_arr, params_collector
