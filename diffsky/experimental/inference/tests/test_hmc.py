""" """

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jran

from .. import hmc

MockParams = namedtuple("MockParams", ("a", "b"))


def test_run_chains_stacks_sequential_outputs_along_chain_axis():
    @jax.jit
    def add_one(x):
        return x + 1.0

    num_chains = 3
    args = (jnp.arange(6.0).reshape(num_chains, 2),)

    out = hmc.run_chains(add_one, args, num_chains)

    assert np.allclose(out, jnp.arange(6.0).reshape(num_chains, 2) + 1.0)


def test_run_warmup_and_sampling_recover_gaussian_posterior():
    ran_key = jran.key(0)
    init_key, warmup_key, sampler_key = jran.split(ran_key, 3)
    target_cov = jnp.array([[0.05, 0.0], [0.0, 0.01]])
    target_stds = jnp.sqrt(jnp.diag(target_cov))
    precision = jnp.linalg.inv(target_cov)
    center_point = jnp.zeros(2)

    def flat_logdensity(x):
        return -0.5 * x @ precision @ x

    num_chains = 2
    hmc_settings = dict(
        warmup_num_steps=50, max_num_doublings=5, target_acceptance_rate=0.8
    )
    chain_inits = hmc.make_chain_inits(
        center_point, num_chains, 2.0, target_cov, init_key
    )
    warmup_states, step_sizes, __, imm = hmc.run_warmup(
        flat_logdensity,
        jran.split(warmup_key, num_chains),
        chain_inits,
        hmc_settings,
        target_cov,
    )

    positions, sample_info = hmc.run_sampling(
        flat_logdensity,
        imm,
        5,
        100,
        jran.split(sampler_key, num_chains),
        warmup_states,
        step_sizes,
    )

    assert positions.shape == (num_chains, 100, 2)
    assert np.all(np.isfinite(positions))
    samples = np.array(positions).reshape(-1, 2)
    assert np.allclose(samples.mean(axis=0), center_point, atol=0.05)
    assert np.allclose(samples.std(axis=0), target_stds, rtol=0.2)
