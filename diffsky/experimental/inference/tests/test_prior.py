""" """

import jax.numpy as jnp
import numpy as np
from jax import random as jran
from jax.flatten_util import ravel_pytree

from ....param_utils import diffsky_param_wrapper_merging as dpwm
from .. import prior, utils

VARIED_PARAM_NAMES = ["frac_quench_cen_x0_tpeak", "frac_quench_cen_k_tpeak"]


def _get_default_uparam_flat():
    return dpwm.DiffskyUParamsFlat(*jnp.zeros(len(dpwm.U_PNAMES_FLAT)))


def test_soft_uniform_log_prior_inside_box_is_near_normalization():
    x = jnp.array([0.5])
    low = jnp.array([0.0])
    high = jnp.array([1.0])
    log_prior = prior._soft_uniform_log_prior(x, low, high)
    assert np.allclose(log_prior, 0.0, atol=1e-3)


def test_soft_uniform_log_prior_penalizes_points_outside_box():
    low = jnp.zeros(3)
    high = jnp.ones(3)
    lp_inside = jnp.sum(
        prior._soft_uniform_log_prior(jnp.array([0.5, 0.5, 0.5]), low, high)
    )
    lp_below = jnp.sum(
        prior._soft_uniform_log_prior(jnp.array([-1.0, 0.5, 0.5]), low, high)
    )
    lp_above = jnp.sum(
        prior._soft_uniform_log_prior(jnp.array([2.0, 0.5, 0.5]), low, high)
    )
    assert lp_below < lp_inside - 10.0
    assert lp_above < lp_inside - 10.0


def test_soft_uniform_log_prior_from_param_coll_defaults_are_finite():
    """Enforce that the default diffsky parameters fall inside the default
    bounds of the soft uniform prior"""
    log_prior = prior.soft_uniform_log_prior_from_param_coll(
        dpwm.DEFAULT_PARAM_COLLECTION
    )
    assert log_prior.shape == (len(dpwm.PNAMES_FLAT),)
    assert np.all(np.isfinite(log_prior))


def test_hard_uniform_prior_returns_in_bounds_samples_of_correct_shape():
    ran_key = jran.key(0)
    n_samples = 100
    low = prior.DEFAULT_LOW_FLAT[:4]
    high = prior.DEFAULT_HIGH_FLAT[:4]

    samples = prior.hard_uniform_prior(ran_key, n_samples, minval=low, maxval=high)

    assert samples.shape == (n_samples, 4)
    assert np.all(samples >= low)
    assert np.all(samples <= high)


def test_sample_from_hard_prior_only_varies_selected_params():
    ran_key = jran.key(0)
    n_samples = 3

    param_coll_list = prior.sample_from_hard_prior(
        ran_key, n_samples, dpwm.DEFAULT_PARAM_COLLECTION, VARIED_PARAM_NAMES
    )

    assert len(param_coll_list) == n_samples
    default_param_flat, _ = ravel_pytree(dpwm.DEFAULT_PARAM_COLLECTION)
    var_idx = np.array([dpwm.PNAMES_FLAT.index(n) for n in VARIED_PARAM_NAMES])
    n_params = len(dpwm.PNAMES_FLAT)
    fixed_idx = np.setdiff1d(np.arange(n_params), var_idx)

    varied_values = []
    for param_coll in param_coll_list:
        param_flat, _ = ravel_pytree(param_coll)
        assert np.allclose(param_flat[fixed_idx], default_param_flat[fixed_idx])
        varied_values.append(param_flat[var_idx])

    varied_values = np.array(varied_values)
    assert np.all(varied_values >= prior.DEFAULT_LOW_FLAT[var_idx])
    assert np.all(varied_values <= prior.DEFAULT_HIGH_FLAT[var_idx])
    assert not np.allclose(varied_values[0], varied_values[1])


def test_flat_logprior_fn_agrees_with_manual_calculation():
    """Enforce that flat_logprior_fn equals the sum of the soft uniform
    log-prior term and the jacobian log-determinant of the bounding function"""
    uparam_flat = _get_default_uparam_flat()
    varied_u_params_list = ["u_" + n for n in VARIED_PARAM_NAMES]
    var_uparam_flat = utils.get_var_param_flat_from_param_flat(
        uparam_flat, varied_u_params_list
    )
    var_flat_idx = utils.compute_varied_params_indices(var_uparam_flat, uparam_flat)

    log_prior = prior.flat_logprior_fn(var_uparam_flat, uparam_flat, var_flat_idx)

    uparam_coll = utils.get_uparam_coll_from_var_uparam_flat(
        var_uparam_flat, uparam_flat
    )
    param_coll = dpwm.get_param_collection_from_u_param_collection(*uparam_coll)
    param_flat, _ = ravel_pytree(param_coll)
    lg_dist_term = jnp.sum(
        prior._soft_uniform_log_prior(
            param_flat[var_flat_idx],
            prior.DEFAULT_LOW_FLAT[var_flat_idx],
            prior.DEFAULT_HIGH_FLAT[var_flat_idx],
        )
    )
    log_abs_det = prior._var_jac_logdet(uparam_coll, var_flat_idx)
    assert np.allclose(log_prior, lg_dist_term + log_abs_det)
