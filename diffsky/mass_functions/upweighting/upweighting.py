import jax.numpy as jnp
import jax.random

from .. import hmf_model, predict_differential_hmf


def downsample_and_upweight(
    lgmp_at_t_obs,
    ult_host_indx,
    z_obs=0.0,
    hmf_params=None,
    randkey=0,
    method="histogram",
    num_hist_bins=30,
    target_nhost=1000,
):
    if hmf_params is None:
        hmf_params = hmf_model.DEFAULT_HMF_PARAMS
    if isinstance(randkey, int):
        randkey = jax.random.key(randkey)
    lgmp_of_host = lgmp_at_t_obs[ult_host_indx]
    is_host = ult_host_indx == jnp.arange(len(ult_host_indx))
    host_lgmp = lgmp_of_host[is_host]
    nhost = is_host.sum()
    assert jnp.all(is_host[ult_host_indx]), "ult_host_indx contains non-hosts"

    if method == "none":
        host_upweights = jnp.ones_like(host_lgmp)

    elif method == "histogram":
        bins = jnp.linspace(host_lgmp.min(), host_lgmp.max(), num_hist_bins)
        bins = jnp.array([bins[0] - 1.0, *bins[1:-1], bins[-1] + 1.0])
        hist = jnp.maximum(jnp.histogram(host_lgmp, bins=bins)[0], 1)
        binned_upweights = (hist * num_hist_bins) / target_nhost
        host_upweights = binned_upweights[jnp.digitize(host_lgmp, bins) - 1]

    elif method == "analytic":
        mass_range = host_lgmp.max() - host_lgmp.min()
        host_dist = predict_differential_hmf(hmf_params, host_lgmp, z_obs)
        host_dist = host_dist / host_dist.sum() * nhost**2
        host_upweights = host_dist * mass_range / target_nhost

    else:
        raise ValueError(f"Unrecognized value for method = {method}")

    host_upweights = jnp.maximum(host_upweights, 1.0)
    r = jax.random.uniform(randkey, (nhost,))
    host_in_sample = r < (1 / host_upweights)

    in_sample = jnp.zeros_like(is_host)
    upweights = jnp.zeros_like(lgmp_of_host)
    in_sample = in_sample.at[is_host].set(host_in_sample)
    upweights = upweights.at[is_host].set(host_upweights)
    in_sample = in_sample[ult_host_indx]
    upweights = upweights[ult_host_indx]
    return in_sample, upweights
