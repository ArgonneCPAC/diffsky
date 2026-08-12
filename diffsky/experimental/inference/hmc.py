import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import time

try:
    import blackjax
    from blackjax.mcmc.nuts import build_kernel, init as nuts_init
    from blackjax.adaptation.window_adaptation import dual_averaging_adaptation
except ImportError as e:
    raise ImportError("This module requires blackjax >= 1.0.") from e


# ---------------------------------------------------------------------------
# 1. Chain initialization
# ---------------------------------------------------------------------------
def make_chain_inits(
    center_point,
    num_chains,
    init_jitter,
    inverse_mass_matrix,
    init_key,
):
    """
    Computes initial position for each chain around center_point (for example the MLE/MAP),
    shape ``(num_chains, num_var_params)``.

    When an inverse mass matrix (e.g. a Fisher covariance) is available, chain starts are draws from
    the Laplace approximation ``N(center_point, init_jitter^2 * imm)``; otherwise isotropic Gaussian
    jitter is used.

    ``init_jitter`` controls over-dispersion and the validity of split-R-hat.
      - ``~2-4`` (recommended for multi-chain diagnostics): chains start *wider*
        than the posterior, so R-hat -> 1 is a real test of convergence rather
        than shared ancestry.
      - ``1`` : draws at the posterior scale (not actually over-dispersed; R-hat
        is a weak diagnostic).
      - ``0`` : all chains start exactly at ``map_flat`` (useful for
        single-chain production to minimize burn-in; meaningless for R-hat).

    Parameters:
        center_point: var_uparam_flat
            Reference point for the initial points to be drawn.
            Can be the MLE/MAP computed with Adam, for example.
        num_chains: int
            Number of chains.
        init_jitter: int
            Controls over-dispersion of the chains initial point.
        inverse_mass_matrix: numpy array
            Can be the IMM precomputed with the Laplace Approximation.
        init_key: random key

    Returns:
        chain_inits: namedtuple with initial positions
            Same structure as ``var_uparam_flat`` (center_point) but with
            num_chains values per parameter.

    """
    map_flat, var_unflatten_fn = ravel_pytree(center_point)
    num_var_params = int(map_flat.shape[0])

    if init_jitter == 0:
        return jnp.broadcast_to(map_flat, (num_chains, num_var_params)).astype(
            map_flat.dtype
        )
    if inverse_mass_matrix is not None:
        chol = jnp.linalg.cholesky(
            init_jitter**2 * inverse_mass_matrix + 1e-12 * jnp.eye(num_var_params)
        )
        noise = jax.random.normal(init_key, (num_chains, num_var_params)) @ chol.T
    else:
        noise = init_jitter * jax.random.normal(init_key, (num_chains, num_var_params))

    chain_inits_flat = map_flat + noise
    chain_inits = jax.vmap(var_unflatten_fn)(chain_inits_flat)

    return chain_inits


# ---------------------------------------------------------------------------
# 2. Warmup
# ---------------------------------------------------------------------------
def _build_stepsize_warmup_fn(
    flat_logdensity,
    inverse_mass_matrix,
    warmup_num_steps,
    max_num_doublings,
    target_accept,
):
    """
    Build a jitted ``warmup_single(rng_key, init_pos, init_ss)`` for the
    Fisher-mass strategy (dual-averaging step-size adaptation, IMM fixed).
    """
    nuts_step = build_kernel()
    da_init, da_update, da_final = dual_averaging_adaptation(target_accept)

    @jax.jit
    def warmup_single(rng_key, init_pos, init_ss):
        state = nuts_init(init_pos, flat_logdensity)
        da_state = da_init(init_ss)

        def one_step(carry, key):
            st, da_st = carry
            ss = jnp.exp(da_st.log_step_size)
            new_st, info = nuts_step(
                key,
                st,
                flat_logdensity,
                ss,
                inverse_mass_matrix,
                max_num_doublings,
            )
            new_da = da_update(da_st, info.acceptance_rate)
            return (new_st, new_da), info

        keys = jax.random.split(rng_key, warmup_num_steps)
        (final_st, final_da), winfo = jax.lax.scan(one_step, (state, da_state), keys)
        step_size = da_final(final_da)
        return final_st, step_size, winfo

    return warmup_single


def _stack_pytree(*trees):
    """
    Stack a list of structurally-identical pytrees along a new leading axis
    (e.g. a list of NUTSInfo -> a NUTSInfo with a chain axis).
    """
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)


def _run_window_warmup_sequential(
    flat_logdensity,
    warmup_keys,
    chain_inits,
    warmup_num_steps,
    max_num_doublings,
    target_accept,
):
    """Stan-style window adaptation (step size + mass matrix), one chain at a
    time. Returns ``(warmup_states, step_sizes, imms, warmup_info)`` where
    ``warmup_states`` / ``imms`` are Python lists (one entry per chain),
    ``step_sizes`` is ``(num_chains,)`` and ``warmup_info`` is the per-chain
    NUTSInfo stacked along a new chain axis."""
    n_chains = len(warmup_keys)

    @jax.jit
    def single_chain_window(rng_key, init_pos):
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            flat_logdensity,
            max_num_doublings=max_num_doublings,
            target_acceptance_rate=target_accept,
        )
        (w_state, parameters), w_info = warmup.run(
            rng_key, init_pos, num_steps=warmup_num_steps
        )
        return w_state, parameters, w_info

    warmup_states, step_sizes, imms, warmup_infos = [], [], [], []
    tree = jax.tree_util
    for c in range(n_chains):
        # chain_inits is a batched varied-param namedtuple; index one chain.
        init_pos_c = tree.tree_map(lambda x, c=c: x[c], chain_inits)
        w_state, parameters, w_info = single_chain_window(warmup_keys[c], init_pos_c)
        warmup_states.append(w_state)
        step_sizes.append(parameters["step_size"])
        imms.append(parameters["inverse_mass_matrix"])
        warmup_infos.append(w_info)
    warmup_info = _stack_pytree(*warmup_infos) if n_chains > 1 else warmup_infos[0]
    return warmup_states, jnp.stack(step_sizes), imms, warmup_info


def run_warmup(
    flat_logdensity,
    warmup_keys,
    chain_inits,
    hmc_settings,
    inverse_mass_matrix,
):
    """Run NUTS warmup and return ``(warmup_states, step_sizes, warmup_info,
    inverse_mass_matrix_for_sampling)``.

    The mode is chosen by whether an ``inverse_mass_matrix`` is provided:

    * provided -> a step-size-only dual-averaging warmup with the mass matrix
      held fixed, run in parallel via :func:`run_chains` (``jax.pmap`` when
      possible). ``inverse_mass_matrix_for_sampling`` is the shared input array.
    * ``None`` -> blackjax ``window_adaptation`` (adapts mass matrix + step
      size), one chain at a time. ``inverse_mass_matrix_for_sampling`` is the
      per-chain Python list returned by the adaptation.
    """
    warmup_num_steps = hmc_settings["warmup_num_steps"]
    max_num_doublings = hmc_settings["max_num_doublings"]
    target_accept = hmc_settings.get("target_acceptance_rate", 0.8)
    initial_step_size = float(hmc_settings.get("initial_step_size", 1.0))
    # chain_inits is a batched varied-param namedtuple (leaves (num_chains,));
    # derive the chain count from the RNG keys split in run_hmc.
    num_chains = len(warmup_keys)

    start = time.time()
    if inverse_mass_matrix is not None:
        warmup_single = _build_stepsize_warmup_fn(
            flat_logdensity,
            inverse_mass_matrix,
            warmup_num_steps,
            max_num_doublings,
            target_accept,
        )
        warmup_states, step_sizes, warmup_info = run_chains(
            warmup_single,
            (
                warmup_keys,
                chain_inits,
                jnp.full((num_chains,), initial_step_size),
            ),
            num_chains=num_chains,
        )
        imm_out = inverse_mass_matrix
    else:
        warmup_states, step_sizes, imm_out, warmup_info = _run_window_warmup_sequential(
            flat_logdensity,
            warmup_keys,
            chain_inits,
            warmup_num_steps,
            max_num_doublings,
            target_accept,
        )

    jax.block_until_ready(step_sizes)
    # The step-size path returns a stacked NUTSInfo (acceptance_rate at the top
    # level); the window path returns a stacked AdaptationInfo with the NUTSInfo
    # nested under `.info`. Normalize so the summary works for both.
    winfo = warmup_info.info if hasattr(warmup_info, "info") else warmup_info
    print(f"warmup elapsed: {time.time() - start:.2f}s")
    print(f"step_size (per chain): {[float(s) for s in step_sizes]}")
    print(f"mean acceptance: {float(jnp.mean(winfo.acceptance_rate)):.4f}")
    print(f"mean leapfrog steps: {float(jnp.mean(winfo.num_integration_steps)):.1f}")
    print(f"divergences: {int(jnp.sum(winfo.is_divergent))}")
    return warmup_states, step_sizes, warmup_info, imm_out


# ---------------------------------------------------------------------------
# 3. Sampling
# ---------------------------------------------------------------------------


def run_chains(chain_fn, args, num_chains):
    """
    Run a single-chain function across `num_chains` chains.

    `args` is a tuple of pytrees, each with a leading axis of size `num_chains`.
    Uses `jax.pmap` (parallel) when `num_chains <= jax.local_device_count()`;
    otherwise falls back to a sequential Python loop, stacking the per-chain
    outputs along a new leading axis. The output pytree shape is identical in
    both cases, so callers do not need to care which path was taken.
    """
    tree = jax.tree_util
    n_devices = jax.local_device_count()

    if num_chains <= n_devices:
        devices = jax.devices()[:num_chains] if num_chains < n_devices else None
        pmapped = jax.pmap(chain_fn, devices=devices, axis_name="chain")
        return pmapped(*args)

    print(
        f"[run_chains] num_chains={num_chains} > num_devices={n_devices}; "
        "running chains sequentially. Set "
        f"XLA_FLAGS='--xla_force_host_platform_device_count={num_chains}' "
        "to enable parallel pmap."
    )
    outputs = []
    for c in range(num_chains):
        args_c = tree.tree_map(lambda x, c=c: x[c], args)
        outputs.append(chain_fn(*args_c))
    return tree.tree_map(lambda *xs: jnp.stack(xs), *outputs)


def run_sampling(
    flat_logdensity,
    inverse_mass_matrix,
    max_num_doublings,
    num_samples,
    sampler_keys,
    warmup_states,
    step_sizes,
):
    """Run NUTS sampling for every chain.

    Returns ``positions`` as a batched varied-param namedtuple (each leaf has
    shape ``(num_chains, num_samples)``) and ``sample_info`` (a NUTSInfo pytree
    with a leading chain axis).

    ``inverse_mass_matrix`` is either a shared array ``(n, n)`` (mass matrix
    was provided or fixed during warmup) -> chains run in parallel via
    :func:`run_chains` (``jax.pmap``); or a Python list of per-chain mass
    matrices (window adaptation) -> chains run sequentially, each with its own.
    """
    num_chains = len(sampler_keys)
    start = time.time()

    if not isinstance(inverse_mass_matrix, list):
        nuts_step = build_kernel()
        _ns = num_samples

        @jax.jit
        def inference_single(rng_key, init_state, ss):
            def one_step(state, key):
                new_st, info = nuts_step(
                    key,
                    state,
                    flat_logdensity,
                    ss,
                    inverse_mass_matrix,
                    max_num_doublings,
                )
                return new_st, (new_st.position, info)

            keys = jax.random.split(rng_key, _ns)
            _, (positions, sample_info) = jax.lax.scan(one_step, init_state, keys)
            return positions, sample_info

        positions, sample_info = run_chains(
            inference_single,
            (sampler_keys, warmup_states, step_sizes),
            num_chains=num_chains,
        )
    else:
        # Per-chain inverse mass matrix + step size; run sequentially.
        nuts_step = build_kernel()

        @jax.jit
        def single_chain_inference(rng_key, init_state, ss, imm):
            def one_step(state, key):
                new_st, info = nuts_step(
                    key,
                    state,
                    flat_logdensity,
                    ss,
                    imm,
                    max_num_doublings,
                )
                return new_st, (new_st.position, info)

            keys = jax.random.split(rng_key, num_samples)
            _, (pos, info) = jax.lax.scan(one_step, init_state, keys)
            return pos, info

        pos_list, info_list = [], []
        for c in range(num_chains):
            pos_c, info_c = single_chain_inference(
                sampler_keys[c], warmup_states[c], step_sizes[c], inverse_mass_matrix[c]
            )
            pos_list.append(pos_c)
            info_list.append(info_c)
        # Stack the per-chain varied-param namedtuples along a new chain axis
        # (leaves -> (num_chains, num_samples)), matching the run_chains output.
        positions = _stack_pytree(*pos_list) if num_chains > 1 else pos_list[0]
        sample_info = _stack_pytree(*info_list) if num_chains > 1 else info_list[0]

    jax.block_until_ready(positions)
    print(f"sampling elapsed: {time.time() - start:.2f}s")
    print(f"mean acceptance: {float(jnp.mean(sample_info.acceptance_rate)):.4f}")
    print(
        f"mean leapfrog steps: {float(jnp.mean(sample_info.num_integration_steps)):.1f}"
    )
    print(f"divergences: {int(jnp.sum(sample_info.is_divergent))}")
    return positions, sample_info
