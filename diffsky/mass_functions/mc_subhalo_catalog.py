"""
"""

import typing

import numpy as np
from diffmah.defaults import DiffmahParams
from diffmah.monte_carlo_diffmah_hiz import mc_diffmah_params_hiz
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.cosmology.flat_wcdm import _age_at_z_kern
from jax import random as jran

from .mc_hosts import mc_host_halos_singlez
from .mc_subs import generate_subhalopop
from .mc_tinfall import mc_time_since_infall


class SubhaloCatalog(typing.NamedTuple):
    mah_params: np.ndarray
    log_mpeak_penultimate_infall: np.ndarray
    log_mpeak_ultimate_infall: np.ndarray
    log_mhost_penultimate_infall: np.ndarray
    log_mhost_ultimate_infall: np.ndarray
    t_obs: np.ndarray
    t_penultimate_infall: np.ndarray
    t_ultimate_infall: np.ndarray
    upids: np.ndarray
    penultimate_indx: np.ndarray
    ultimate_indx: np.ndarray


def mc_subhalo_catalog_singlez(ran_key, lgmp_min, redshift, volume_com):
    """Monte Carlo realization of a subhalo catalog at the input redshift

    Parameters
    ----------
    ran_key : jran.PRNGKey

    lgmp_min : float
        Base-10 log of the halo mass competeness limit of the generated population

        Smaller values of lgmp_min produce more halos in the returned sample

    redshift : float
        Redshift of the halo population

    volume_com : float
        volume_com = Lbox**3 where Lbox is in comoving in units of Mpc/h

        Larger values of volume_com produce more halos in the returned sample

    Returns
    -------
    SubhaloCatalog : namedtuple

        mah_params: np.ndarray, shape (n_halos, 4)
        log_mpeak_penultimate_infall: np.ndarray, shape (n_halos, )
        log_mpeak_ultimate_infall: np.ndarray, shape (n_halos, )
        log_mhost_penultimate_infall: np.ndarray, shape (n_halos, )
        log_mhost_ultimate_infall: np.ndarray, shape (n_halos, )
        t_obs: float
        t_penultimate_infall: np.ndarray, shape (n_halos, )
        t_ultimate_infall: np.ndarray, shape (n_halos, )
        upids: np.ndarray, shape (n_halos, )
        penultimate_indx: np.ndarray, shape (n_halos, )
        ultimate_indx: np.ndarray, shape (n_halos, )

    """

    host_key1, host_key2, sub_key1, sub_key2 = jran.split(ran_key, 4)
    hosts_logmh_at_z = mc_host_halos_singlez(host_key1, lgmp_min, redshift, volume_com)
    hosts_halo_id = np.arange(hosts_logmh_at_z.size).astype(int)

    subhalo_info = generate_subhalopop(sub_key1, hosts_logmh_at_z, lgmp_min)
    subs_lgmu, subs_lgmhost, subs_host_halo_indx = subhalo_info

    subs_logmh_at_z = subs_lgmu + subs_lgmhost

    t_obs = _age_at_z_kern(redshift, *DEFAULT_COSMOLOGY)

    subs_time_since_infall = mc_time_since_infall(subs_lgmu, t_obs)

    hosts_diffmah = DiffmahParams(
        *mc_diffmah_params_hiz(host_key2, t_obs, hosts_logmh_at_z)
    )
    subs_diffmah = DiffmahParams(
        *mc_diffmah_params_hiz(sub_key2, t_obs, subs_logmh_at_z)
    )

    n_cens = hosts_logmh_at_z.size

    mah_params = DiffmahParams(
        *[np.concatenate((x, y)) for x, y in zip(hosts_diffmah, subs_diffmah)]
    )

    log_mpeak_penultimate_infall = np.concatenate((hosts_logmh_at_z, subs_logmh_at_z))
    log_mpeak_ultimate_infall = np.concatenate((hosts_logmh_at_z, subs_logmh_at_z))

    log_mhost_penultimate_infall = np.concatenate((hosts_logmh_at_z, subs_lgmhost))
    log_mhost_ultimate_infall = np.concatenate((hosts_logmh_at_z, subs_lgmhost))

    t_penultimate_infall = np.concatenate(
        (np.zeros(n_cens) + t_obs, t_obs - subs_time_since_infall)
    )
    t_ultimate_infall = np.concatenate(
        (np.zeros(n_cens) + t_obs, t_obs - subs_time_since_infall)
    )

    upids = np.concatenate((np.zeros(n_cens) - 1.0, hosts_halo_id[subs_host_halo_indx]))

    penultimate_indx = np.concatenate(
        (np.arange(n_cens).astype(int), subs_host_halo_indx)
    )
    ultimate_indx = np.concatenate((np.arange(n_cens).astype(int), subs_host_halo_indx))

    subcat = SubhaloCatalog(
        mah_params,
        log_mpeak_penultimate_infall,
        log_mpeak_ultimate_infall,
        log_mhost_penultimate_infall,
        log_mhost_ultimate_infall,
        t_obs,
        t_penultimate_infall,
        t_ultimate_infall,
        upids,
        penultimate_indx,
        ultimate_indx,
    )
    return subcat
