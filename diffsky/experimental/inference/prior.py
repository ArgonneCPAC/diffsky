import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from diffsky.param_utils import diffsky_param_wrapper_merging as dpwm

from . import utils

_lo_coll, _hi_coll = utils.unpack_nested_samples(dpwm.BOUND_PARAM_COLLECTION)
DEFAULT_LOW_FLAT = jnp.array(dpwm.unroll_param_collection_into_flat_array(*_lo_coll))
DEFAULT_HIGH_FLAT = jnp.array(dpwm.unroll_param_collection_into_flat_array(*_hi_coll))


# Unbounding functions
f = dpwm.get_param_collection_from_u_param_collection


def soft_uniform_log_prior_from_param_coll(
    param_coll, low=DEFAULT_LOW_FLAT, high=DEFAULT_HIGH_FLAT, k=20.0
):
    """
    Wrapper around _soft_uniform_log_prior.
    Accepts diffsky's ParamCollection namedtuple.
    """
    # param_flat = jnp.array(dpwm.unroll_param_collection_into_flat_array(*param_coll))
    param_flat, _ = ravel_pytree(param_coll)
    return _soft_uniform_log_prior(param_flat, low, high, k)


def _soft_uniform_log_prior(
    param_flat, low=DEFAULT_LOW_FLAT, high=DEFAULT_HIGH_FLAT, k=20.0
):
    """
    Differentiable approximation of the uniform log-prior.
    Defined on bounded space. Accepts arrays.
    """
    # log(sigmoid(z)) is equivalent to -softplus(-z)
    # This approximates the 'box' shape of the uniform PDF
    log_prob = -jax.nn.softplus(-k * (param_flat - low)) - jax.nn.softplus(
        -k * (high - param_flat)
    )

    # Normalization constant (1 / (high - low))
    log_normalization = jnp.log(high - low)

    return log_prob - log_normalization


def _var_jac_logdet(uparam_coll, var_flat_idx):
    """
    log|det df/du| restricted to the var parameters.
    Defines internally the flat version of the bounding function f.
    """

    # Flat version of f
    uparam_flat, uphi_fn = ravel_pytree(uparam_coll)

    def f_flat(uparam_flat):
        # param_flat = \phi( param_coll = f( \phi^*(uparam_flat) ) )
        return ravel_pytree(f(*uphi_fn(uparam_flat)))[0]

    # Mask all parameters that are not varied
    tangent = jnp.zeros_like(uparam_flat).at[var_flat_idx].set(1.0)
    # Compute jacobian of the f_flat function + mask fixed params
    _, jvp_out = jax.jvp(f_flat, (uparam_flat,), (tangent,))

    # Get only var params. diagonal terms
    diag = jvp_out[var_flat_idx]
    return jnp.sum(jnp.log(jnp.abs(diag) + 1e-30))


# Hard prior on bounded space


def hard_uniform_prior(
    ran_key,
    n_samples,
    minval=DEFAULT_LOW_FLAT,
    maxval=DEFAULT_HIGH_FLAT,
):
    """
    Return samples from hard uniform prior.
    This maybe more convenient than sampling from the soft prior with MCMC for certain applications.
    Defined on bounded space.
    """
    return jax.random.uniform(
        ran_key,
        shape=(n_samples, len(minval)),
        minval=minval,
        maxval=maxval,
    )


def sample_from_hard_prior(ran_key, n_samples, param_coll, var_params_list):
    """
    Returns a set of copies of param_coll replaced with the prior samples only for the parameters in var_params_list.

    Remember: var_param_flat is handled with ravel_pytree + var_flat_idx.
    """

    param_flat = dpwm.unroll_param_collection_into_flat_array(*param_coll)
    var_param_flat = utils.get_var_param_flat_from_param_flat(
        param_flat, var_params_list
    )
    var_flat_idx = utils.compute_varied_params_indices(var_param_flat, param_flat)
    _, unravel_fn = ravel_pytree(var_param_flat)

    samples = hard_uniform_prior(
        ran_key,
        n_samples,
        minval=DEFAULT_LOW_FLAT[var_flat_idx],
        maxval=DEFAULT_HIGH_FLAT[var_flat_idx],
    )

    param_coll_list = [
        utils.get_param_coll_from_var_param_flat(unravel_fn(s), param_flat)
        for s in samples
    ]

    return param_coll_list


# -- Flat function --


def flat_logprior_fn(var_uparam_flat, diffsky_params, var_flat_idx):
    """
    diffsky_params is uparam_flat containing all diffsky parameters.

    Log-prior as a function of the flat var vector.
    Defined in the unbounded space.
    """
    # \theta* from \theta-*_var
    uparam_coll = utils.get_uparam_coll_from_var_uparam_flat(
        var_uparam_flat, diffsky_params
    )
    # \theta from \theta*
    param_coll = f(*uparam_coll)

    # first term
    param_flat, _ = ravel_pytree(param_coll)
    lg_dist_term = jnp.sum(
        _soft_uniform_log_prior(
            param_flat[var_flat_idx],
            DEFAULT_LOW_FLAT[var_flat_idx],
            DEFAULT_HIGH_FLAT[var_flat_idx],
        )
    )
    # second term
    log_abs_det = _var_jac_logdet(uparam_coll, var_flat_idx)

    return lg_dist_term + log_abs_det
