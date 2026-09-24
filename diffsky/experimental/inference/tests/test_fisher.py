""" """

import jax.numpy as jnp
import numpy as np
from jax import random as jran

from .. import fisher


def test_compute_fisher_matrix_recovers_analytic_gaussian():
    """Enforce that compute_fisher_matrix recovers the known precision matrix
    of a gaussian log-likelihood plus gaussian log-prior"""
    likelihood_fim = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    prior_precision = 0.7
    eval_point = jnp.zeros(2)

    def flat_loglikelihood(x):
        return -0.5 * x @ likelihood_fim @ x

    def flat_logprior(x):
        return -0.5 * prior_precision * x @ x

    res = fisher.compute_fisher_matrix(
        eval_point, flat_loglikelihood, flat_logprior, verbose=False
    )

    expected_fim = likelihood_fim + prior_precision * jnp.eye(2)
    assert np.allclose(res.fim, expected_fim, rtol=1e-4)
    assert np.allclose(res.prior_fim, prior_precision * jnp.eye(2), rtol=1e-4)
    assert np.allclose(res.covariance_matrix, np.linalg.inv(expected_fim), rtol=1e-4)


def test_compute_fisher_matrix_fails_when_no_prior_is_given():
    def flat_loglikelihood(x):
        return -0.5 * x @ x

    try:
        fisher.compute_fisher_matrix(
            jnp.zeros(2), flat_loglikelihood, None, include_prior=True, verbose=False
        )
        raise NameError("compute_fisher_matrix should require flat_logprior")
    except ValueError:
        pass
