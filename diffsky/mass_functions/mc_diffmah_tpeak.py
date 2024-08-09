"""
"""

import typing

import numpy as np
from diffmah.diffmah_kernels import DiffmahParams, _log_mah_kern
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
    halo_ids: np.ndarray
    mah_params: np.ndarray
    lgmp_pen_inf: np.ndarray
    lgmp_ult_inf: np.ndarray
    lgmhost_pen_inf: np.ndarray
    lgmhost_ult_inf: np.ndarray
    t_obs: np.ndarray
    t_pen_inf: np.ndarray
    t_ult_inf: np.ndarray
    upids: np.ndarray
    pen_indx: np.ndarray
    ult_indx: np.ndarray


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
        lgmp_pen_inf: np.ndarray, shape (n_halos, )
        lgmp_ult_inf: np.ndarray, shape (n_halos, )
        lgmhost_pen_inf: np.ndarray, shape (n_halos, )
        lgmhost_ult_inf: np.ndarray, shape (n_halos, )
        t_obs: float
        t_pen_inf: np.ndarray, shape (n_halos, )
        t_ult_inf: np.ndarray, shape (n_halos, )
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
    halo_ids = np.arange(n_cens + n_sats).astype(int)
    subs_diffmah, subs_t_peak = mc_diffmah_params_satpop(
        diffmahpop_params, subs_logmh_at_z, t_obs + _ZS, sub_key2
    )

    # For every sub, get diffmah params of its host halo
    subs_host_diffmah = DiffmahParams(*[x[subs_host_halo_indx] for x in hosts_diffmah])
    subs_host_t_peak = hosts_t_peak[subs_host_halo_indx]

    mah_params = DiffmahParams(
        *[np.concatenate((x, y)) for x, y in zip(hosts_diffmah, subs_diffmah)]
    )

    subs_lgmp_pen_inf = _log_mah_kern(subs_diffmah, subs_t_peak, subs_t_peak, lgt0)
    subs_lgmp_ult_inf = subs_lgmp_pen_inf

    subs_lgmhost_pen_inf = _log_mah_kern(
        subs_host_diffmah, subs_t_peak, subs_host_t_peak, lgt0
    )
    subs_lgmhost_ult_inf = subs_lgmhost_pen_inf

    lgmp_pen_inf = np.concatenate((hosts_logmh_at_z, subs_lgmp_pen_inf))
    lgmp_ult_inf = np.concatenate((hosts_logmh_at_z, subs_lgmp_ult_inf))

    lgmhost_pen_inf = np.concatenate((hosts_logmh_at_z, subs_lgmhost_pen_inf))
    lgmhost_ult_inf = np.concatenate((hosts_logmh_at_z, subs_lgmhost_ult_inf))

    t_pen_inf = np.concatenate((hosts_t_peak, subs_t_peak))
    t_ult_inf = np.concatenate((hosts_t_peak, subs_t_peak))

    upids = np.concatenate(
        (np.zeros(n_cens).astype(int) - 1, hosts_halo_id[subs_host_halo_indx])
    )

    penultimate_indx = np.concatenate(
        (np.arange(n_cens).astype(int), subs_host_halo_indx)
    )
    ultimate_indx = np.concatenate((np.arange(n_cens).astype(int), subs_host_halo_indx))
    subcat = SubhaloCatalog(
        halo_ids,
        mah_params,
        lgmp_pen_inf,
        lgmp_ult_inf,
        lgmhost_pen_inf,
        lgmhost_ult_inf,
        t_obs,
        t_pen_inf,
        t_ult_inf,
        upids,
        penultimate_indx,
        ultimate_indx,
    )
    return subcat
