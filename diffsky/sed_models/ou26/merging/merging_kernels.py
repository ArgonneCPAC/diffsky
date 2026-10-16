""" """

from jax import jit as jjit
from jax import numpy as jnp

MC_P_MERGE_MAX = 1.0 - 1e-6


@jjit
def compute_x_tot_from_x_in_situ(
    x_in_situ, p_merge, sat_weights, halo_indx, frac_receive=1.0
):
    """Compute quantity X after including effects from merging

    Parameters
    ----------
    x_in_situ : array of shape (n, )

    p_merge : array of shape (n, )
        Probability that the object merges into its associated central
        0<=p_merge<=1, with p_merge=0 for centrals

    sat_weights : array of shape (n, )
        Multiplicity factor for subhalos
        Equals 1 for host halos, and <Nsub | Mhost> for subhalos

    halo_indx : array of shape (n, )
        Index to deposit quantity X

    frac_receive : float or array of shape (n, ), optional
        Fraction of X that the associated central receives from a merged satellite
        frac_receive < 1 ==> X is not conserved

        Example use-case: satellite emission lines may be partially or entirely
            extinguished upon merging, in which case merged satellites will still vanish
            but their associated central will not receive boosted emission line flux

    Returns
    -------
    x_tot : array, shape (n, )

    """
    ngals = x_in_situ.shape[0]
    indx_to_keep = jnp.arange(ngals).astype("i8")

    merge_weight = p_merge * sat_weights

    x_to_keep = x_in_situ * (1 - p_merge)
    x_to_receive = x_in_situ * merge_weight * frac_receive

    x_tot = jnp.zeros_like(x_in_situ)
    x_tot = x_tot.at[halo_indx].add(x_to_receive)
    x_tot = x_tot.at[indx_to_keep].add(x_to_keep)

    return x_tot


@jjit
def get_mc_p_merge(uran, p_merge):
    mc_p_merge = jnp.where(uran < p_merge, MC_P_MERGE_MAX, 0.0)
    return mc_p_merge


@jjit
def get_in_plus_ex_situ_ssp_weights(
    mstar_in_situ, ssp_weights_in_situ, p_merge, sat_weights, halo_indx
):
    x_in_situ = ssp_weights_in_situ * mstar_in_situ.reshape((-1, 1, 1))
    x_tot = compute_x_tot_from_x_in_situ(
        x_in_situ,
        p_merge[:, jnp.newaxis, jnp.newaxis],
        sat_weights[:, jnp.newaxis, jnp.newaxis],
        halo_indx,
    )
    norm = jnp.sum(x_tot, axis=(1, 2))
    ssp_weights = x_tot / norm.reshape((-1, 1, 1))
    return ssp_weights
