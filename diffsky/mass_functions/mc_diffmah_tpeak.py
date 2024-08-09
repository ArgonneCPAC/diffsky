"""
"""

import typing

import numpy as np
from diffmah.diffmah_kernels import DiffmahParams
from diffmah.diffmahpop_kernels.diffmahpop_params_monocensat import (
    DEFAULT_DIFFMAHPOP_PARAMS,
)
from diffmah.diffmahpop_kernels.mc_diffmahpop_kernels_monocens import (
    mc_diffmah_params_cenpop,
)
from diffmah.diffmahpop_kernels.mc_diffmahpop_kernels_monosats import (
    mc_diffmah_params_satpop,
)
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.cosmology.flat_wcdm import _age_at_z_kern
from jax import random as jran

from .mc_hosts import mc_host_halos_singlez
from .mc_subs import generate_subhalopop


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


def mc_subhalo_catalog_singlez(
    ran_key,
    lgmp_min,
    redshift,
    volume_com,
    cosmo=DEFAULT_COSMOLOGY,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
):
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

    subhalo_info = generate_subhalopop(sub_key1, hosts_logmh_at_z, lgmp_min)
    subs_lgmu, subs_lgmhost, subs_host_halo_indx = subhalo_info

    subs_logmh_at_z = subs_lgmu + subs_lgmhost

    t_obs = _age_at_z_kern(redshift, *cosmo)
    t_0 = _age_at_z_kern(0.0, *cosmo)
    lgt0 = np.log10(t_0)

    n_cens = hosts_logmh_at_z.size
    hosts_halo_id = np.arange(n_cens).astype(int)
    _ZH = np.zeros(n_cens)

    hosts_diffmah, hosts_t_peak = mc_diffmah_params_cenpop(
        diffmahpop_params, hosts_logmh_at_z, t_obs + _ZH, host_key2, lgt0
    )

    n_sats = subs_logmh_at_z.size
    _ZS = np.zeros(n_sats)
    subs_diffmah, subs_t_peak = mc_diffmah_params_satpop(
        diffmahpop_params, subs_logmh_at_z, t_obs + _ZS, sub_key2
    )

    mah_params = DiffmahParams(
        *[np.concatenate((x, y)) for x, y in zip(hosts_diffmah, subs_diffmah)]
    )

    log_mpeak_penultimate_infall = np.concatenate((hosts_logmh_at_z, subs_logmh_at_z))
    log_mpeak_ultimate_infall = np.concatenate((hosts_logmh_at_z, subs_logmh_at_z))

    log_mhost_penultimate_infall = np.concatenate((hosts_logmh_at_z, subs_lgmhost))
    log_mhost_ultimate_infall = np.concatenate((hosts_logmh_at_z, subs_lgmhost))

    t_penultimate_infall = np.concatenate((hosts_t_peak, subs_t_peak))
    t_ultimate_infall = np.concatenate((hosts_t_peak, subs_t_peak))

    upids = np.concatenate(
        (np.zeros(n_cens).astype(int) - 1, hosts_halo_id[subs_host_halo_indx])
    )

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
