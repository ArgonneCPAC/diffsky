import time
from collections import namedtuple

import jax
import jax.numpy as jnp

FisherMatrixOut = namedtuple(
    "FisherMatrixOut", ("fim", "covariance_matrix", "prior_fim")
)


def compute_fisher_matrix(
    eval_point,
    flat_loglikelihood,
    flat_logprior=None,
    include_prior=True,
    verbose=True,
    **kwargs_likelihood,
):
    """
    Compute the posterior Fisher matrix at ``eval_point``.

    Parameters
    ----------
    flat_loglikelihood : callable
        Log-likelihood as a function of var_uparam_flat.
    flat_eval_point : var_uparam_flat, input to flat_loglikelihood
        Point at which to evaluate the Fisher (should be the MLE / MAP).
    flat_logprior : callable or None
        Log-prior as a function of var_uparam_flat.
        Required when *include_prior* is True.
    include_prior : bool
        Whether to add the prior Fisher.
        Currently the pipeline expects the prior.
    verbose : bool
        Print per-column progress and timing.

    Returns
    -------
    FisherMatrixOut
        Namedtuple with fields:
        - ``fim``: (n, n) Fisher information matrix.
        - ``covariance_matrix``: (n, n) inverse of fim.
        - ``prior_fim``: (n, n) prior Fisher if computed, else None.
    """
    flat_grad = jax.grad(flat_loglikelihood)
    num_var_params = len(eval_point)

    @jax.jit
    def hessian_column(p, e_i):
        return jax.grad(
            lambda x, **kwargs: jnp.vdot(jnp.asarray(flat_grad(x, **kwargs)), e_i)
        )(p, **kwargs_likelihood)

    eye = jnp.eye(num_var_params)

    if verbose:
        print("computing likelihood fisher matrix column-by-column")
    start = time.time()
    cols = []
    for i in range(num_var_params):
        col_i = jnp.asarray(hessian_column(eval_point, eye[i]))
        jax.block_until_ready(col_i)
        cols.append(col_i)
        if verbose:
            print(
                f"  column {i + 1}/{num_var_params} done ({time.time() - start:.1f}s)"
            )

    likelihood_hessian = jnp.stack(cols, axis=1)
    fim = -likelihood_hessian

    prior_fim = None
    if include_prior:
        if flat_logprior is None:
            raise ValueError("flat_logprior must be provided when include_prior=True")
        if verbose:
            print("computing prior fisher matrix")
        prior_hessian = jnp.asarray(jax.hessian(flat_logprior)(eval_point))
        prior_fim = -prior_hessian
        fim = fim + prior_fim

    if verbose:
        print("inverting fisher matrix -> covariance matrix")
    covariance_matrix = jnp.linalg.inv(fim)
    jax.block_until_ready(covariance_matrix)

    if verbose:
        print(f"fisher + inversion elapsed: {time.time() - start:.2f}s")

    return FisherMatrixOut(
        fim=fim, covariance_matrix=covariance_matrix, prior_fim=prior_fim
    )


def sample_from_gaussian(ran_key, eval_point, cov, num_samples):
    samples = jax.random.multivariate_normal(
        ran_key, mean=jnp.asarray(eval_point), cov=cov, shape=int(num_samples)
    )
    return samples
