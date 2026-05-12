"""The generate_subhalopop function generates a Monte Carlo realization of a subhalo
population defined by its cumulative conditional subhalo mass function, CCSHMF.
Starting with a simulated snapshot or lightcone with only host halos,
generate_subhalopop can be used to add subhalos with synthetic values of Mpeak.

"""

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .ccshmf_model import DEFAULT_CCSHMF_PARAMS, predict_ccshmf
from .kernels.ccshmf_kernels import DEFAULT_CCSHMF_KERN_PARAMS, lg_ccshmf_kern
from .measure_ccshmf import get_lgmu_cutoff

N_LGMU_TABLE = 100
U_TABLE = np.linspace(1, 0, N_LGMU_TABLE)


def mc_generate_subhalopop_singlehalo(
    ran_key, lgmu_table, ntot, ccshmf_kern_params=DEFAULT_CCSHMF_KERN_PARAMS
):
    uran = jran.uniform(ran_key, minval=0, maxval=1, shape=(ntot,))
    cdf_counts = 10 ** lg_ccshmf_kern(ccshmf_kern_params, lgmu_table)
    cdf_counts = cdf_counts - cdf_counts[0]
    cdf_counts = cdf_counts / cdf_counts[-1]

    mc_lg_mu = np.interp(uran, cdf_counts, lgmu_table)

    return mc_lg_mu


@jjit
def generate_subhalopop_kern(
    uran, lgmhost, lgmp_min, ccshmf_params=DEFAULT_CCSHMF_PARAMS
):
    lgmu_cutoff = get_lgmu_cutoff(lgmhost, lgmp_min, 1)
    lgmu_table = U_TABLE * lgmu_cutoff
    cdf_counts = 10 ** predict_ccshmf(ccshmf_params, lgmhost, lgmu_table)
    cdf_counts = cdf_counts - cdf_counts[0]
    cdf_counts = cdf_counts / cdf_counts[-1]

    mc_lg_mu = jnp.interp(uran, cdf_counts, lgmu_table)

    return mc_lg_mu


_A = (0, 0, None, None)
generate_subhalopop_vmap = jjit(vmap(generate_subhalopop_kern, in_axes=_A))


def generate_subhalopop(
    ran_key, lgmhost_arr, lgmp_min, ccshmf_params=DEFAULT_CCSHMF_PARAMS
):
    """Generate a population of subhalos with synthetic values of Mpeak

    Parameters
    ----------
    ran_key : jax.random.PRNGKey

    lgmhost_arr : ndarray of shape (nhosts, )
        Base-10 log of host halo mass

    lgmp_min : float
        Base-10 log of the smallest Mpeak value of the synthetic subhalos

    cshmf_params : namedtuple, optional
        parameters of the CCSHMF

    Returns
    -------
    mc_lg_mu : ndarray of shape (nsubs, )
        Base-10 log of μ=Msub/Mhost of the Monte Carlo subhalo population

    lgmhost_pop : ndarray of shape (nsubs, )
        Base-10 log of Mhost of the Monte Carlo subhalo population

    host_halo_indx : ndarray of shape (nsubs, )
        Index of the input host halo of each generated subhalo,
        so that lgmhost_pop = lgmhost_arr[host_halo_indx].
        Thus all values satisfy 0 <= host_halo_indx < nhosts

    """
    mean_counts = _compute_mean_subhalo_counts(lgmhost_arr, lgmp_min)
    uran_key, counts_key = jran.split(ran_key, 2)
    subhalo_counts_per_halo = jran.poisson(counts_key, mean_counts)
    ntot = jnp.sum(subhalo_counts_per_halo)
    urandoms = jran.uniform(uran_key, shape=(ntot,))
    lgmhost_pop = np.repeat(lgmhost_arr, subhalo_counts_per_halo)
    halo_ids = np.arange(lgmhost_arr.size).astype(int)
    host_halo_indx = np.repeat(halo_ids, subhalo_counts_per_halo)
    mc_lg_mu = generate_subhalopop_vmap(urandoms, lgmhost_pop, lgmp_min, ccshmf_params)
    return mc_lg_mu, lgmhost_pop, host_halo_indx


def _compute_mean_subhalo_counts(
    lgmhost, lgmp_min, ccshmf_params=DEFAULT_CCSHMF_PARAMS
):
    lgmu_cutoff = get_lgmu_cutoff(lgmhost, lgmp_min, 1)
    mean_counts = 10 ** predict_ccshmf(ccshmf_params, lgmhost, lgmu_cutoff)
    return mean_counts
