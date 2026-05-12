""" """

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from .hmf_model import DEFAULT_HMF_PARAMS, predict_cuml_hmf

N_LGMU_TABLE = 200
U_TABLE = jnp.linspace(0, 1, N_LGMU_TABLE)
LGMH_MAX = 17.0


@jjit
def _compute_nhalos_tot(hmf_params, lgmp_min, redshift, volume_com_mpc):
    nhalos_per_mpc3 = 10 ** predict_cuml_hmf(hmf_params, lgmp_min, redshift)
    nhalos_tot = nhalos_per_mpc3 * volume_com_mpc
    return nhalos_tot


@jjit
def _get_hmf_cdf_interp_tables(hmf_params, lgmp_min, redshift, lgmp_max=LGMH_MAX):
    dlgmp = lgmp_max - lgmp_min
    lgmp_table = U_TABLE * dlgmp + lgmp_min

    cdf_table = 10 ** predict_cuml_hmf(hmf_params, lgmp_table, redshift)
    cdf_table = cdf_table - cdf_table[0]
    cdf_table = cdf_table / cdf_table[-1]

    return lgmp_table, cdf_table


@jjit
def _mc_host_halos_singlez_kern(
    uran, hmf_params, lgmp_min, redshift, lgmp_max=LGMH_MAX
):
    lgmp_table, cdf_table = _get_hmf_cdf_interp_tables(
        hmf_params, lgmp_min, redshift, lgmp_max=lgmp_max
    )
    mc_lg_mp = jnp.interp(uran, cdf_table, lgmp_table)
    return mc_lg_mp


def mc_host_halos_singlez(
    ran_key,
    lgmp_min,
    redshift,
    volume_com_mpc,
    hmf_params=DEFAULT_HMF_PARAMS,
    lgmp_max=LGMH_MAX,
):
    """Monte Carlo realization of the host halo mass function at the input redshift

    Parameters
    ----------
    ran_key : jran.PRNGKey

    lgmp_min : float
        Base-10 log of the halo mass competeness limit of the generated population
        Halo mass is in units of Msun (not Msun/h)

        Smaller values of lgmp_min produce more halos in the returned sample

    redshift : float
        Redshift of the halo population

    volume_com_mpc : float
        Comoving volume of the generated population in units of Mpc^3

        Larger values of volume_com produce more halos in the returned sample

    Returns
    -------
    lgmp_halopop : ndarray, shape (n_halos, )
        Base-10 log of the halo mass of the generated population

    Notes
    -----
    Note that both number density and halo mass are defined in
    physical units (not h=1 units)

    """
    counts_key, u_key = jran.split(ran_key, 2)
    mean_nhalos = _compute_nhalos_tot(hmf_params, lgmp_min, redshift, volume_com_mpc)
    mean_nhalos_lgmax = _compute_nhalos_tot(
        hmf_params, lgmp_max, redshift, volume_com_mpc
    )
    mean_nhalos = mean_nhalos - mean_nhalos_lgmax

    nhalos = jran.poisson(counts_key, mean_nhalos)
    uran = jran.uniform(u_key, minval=0, maxval=1, shape=(nhalos,))
    lgmp_halopop = _mc_host_halos_singlez_kern(
        uran, hmf_params, lgmp_min, redshift, lgmp_max=lgmp_max
    )
    return np.array(lgmp_halopop)
