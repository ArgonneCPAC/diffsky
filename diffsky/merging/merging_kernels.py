""" """

from jax import jit as jjit
from jax import numpy as jnp


@jjit
def compute_x_tot_from_x_in_situ(
    x_in_situ, p_merge, nsat_weights, halo_indx, frac_receive=1.0
):
    """Compute quantity X after including effects from merging

    Parameters
    ----------
    x_in_situ : array of shape (n, )

    p_merge : array of shape (n, )
        Probability that the object merges into its associated central
        0<=p_merge<=1, with p_merge=0 for centrals

    nsat_weights : array of shape (n, )
        Multiplicity factor for satellites

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

    merge_weight = p_merge * nsat_weights

    x_to_keep = x_in_situ * (1 - p_merge)
    x_to_receive = x_in_situ * merge_weight * frac_receive

    x_tot = jnp.zeros_like(x_in_situ)
    x_tot = x_tot.at[halo_indx].add(x_to_receive)
    x_tot = x_tot.at[indx_to_keep].add(x_to_keep)

    return x_tot
