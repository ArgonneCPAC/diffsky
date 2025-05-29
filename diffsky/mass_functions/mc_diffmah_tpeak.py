""" """

from collections import namedtuple

import numpy as np
from diffmah.diffmah_kernels import DiffmahParams, _log_mah_kern
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_cenpop
from diffmah.diffmahpop_kernels.mc_bimod_sats import mc_satpop
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.cosmology.flat_wcdm import _age_at_z_kern, age_at_z0
from jax import random as jran

from .mc_hosts import mc_host_halos_singlez
from .mc_subs import generate_subhalopop

_SUBCAT_KEYS = (
    "halo_ids",
    "mah_params",
    "host_mah_params",
    "logmp0",
    "logmp_t_obs",
    "logmp_pen_inf",
    "logmp_ult_inf",
    "logmhost_pen_inf",
    "logmhost_ult_inf",
    "t_obs",
    "t_pen_inf",
    "t_ult_inf",
    "upids",
    "pen_host_indx",
    "ult_host_indx",
)
SubhaloCatalog = namedtuple("SubhaloCatalog", _SUBCAT_KEYS)


def mc_subhalos(
    ran_key,
    z_obs,
    lgmp_min,
    volume_com=None,
    hosts_logmh_at_z=None,
    cosmo_params=DEFAULT_COSMOLOGY,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
):
    """Monte Carlo realization of a subhalo catalog at a single redshift

    Parameters
    ----------
    ran_key : jran.PRNGKey

    lgmp_min : float
        Base-10 log of the halo mass competeness limit of the generated population
        Smaller values of lgmp_min produce more halos in the returned sample
        A small fraction of halos will have slightly smaller masses than lgmp_min

    z_obs : float
        Redshift of the halo population

    volume_com : float, optional
        volume_com = Lbox**3 where Lbox is in comoving in units of Mpc/h
        Default is None, in which case argument hosts_logmh_at_z must be passed

        Larger values of volume_com produce more halos in the returned sample

    hosts_logmh_at_z : ndarray, optional
        Grid of host halo masses at the input redshift.
        Default is None, in which case volume_com argument must be passed
        and the host halo mass function will be randomly sampled.

    Returns
    -------
    SubhaloCatalog : namedtuple

        halo_ids: array, shape (n_halos, )

        mah_params: array, shape (n_halos, 4)

        t_peak: array, shape (n_halos, s)

        host_mah_params: array, shape (n_halos, 4)
            mah_params of the host halo of every object
            For host halos, host_mah_params==mah_params

        host_t_peak: array, shape (n_halos, s)
            t_peak of the host halo of every object
            For host halos, host_t_peak==t_peak

        lgmp_pen_inf: array, shape (n_halos, )
            (sub)halo mass at t_pen_inf

        lgmp_ult_inf: array, shape (n_halos, )
            (sub)halo mass at t_ult_inf

        lgmhost_pen_inf: array, shape (n_halos, )
            Host halo mass at t_pen_inf.
            For host halos, lgmhost_pen_inf = lgmp_pen_inf.
            For subs, lgmhost_pen_inf is computed from the diffmah params of the host.

        lgmhost_ult_inf: array, shape (n_halos, )
            Host halo mass at t_ult_inf.
            For host halos, lgmhost_ult_inf = lgmp_ult_inf.
            For subs, lgmhost_ult_inf is computed from the diffmah params of the host.

        t_obs: float

        t_pen_inf: array, shape (n_halos, )
            Set equal to t_peak for both hosts and subs

        t_ult_inf: array, shape (n_halos, )
            Set equal to t_peak for both hosts and subs

        upids: array, shape (n_halos, )

        pen_indx: array, shape (n_halos, )
            Index of the halo hosting the object at t_pen_inf

        ult_indx: array, shape (n_halos, )
            Index of the halo hosting the object at t_ult_inf

    """

    host_key1, host_key2, sub_key1, sub_key2 = jran.split(ran_key, 4)
    if hosts_logmh_at_z is None:
        msg = "Must pass volume_com argument if not passing hosts_logmh_at_z"
        assert volume_com is not None, msg
        hosts_logmh_at_z = mc_host_halos_singlez(host_key1, lgmp_min, z_obs, volume_com)

    subhalo_info = generate_subhalopop(sub_key1, hosts_logmh_at_z, lgmp_min)
    subs_lgmu, subs_lgmhost, subs_host_halo_indx = subhalo_info

    subs_logmh_at_z = subs_lgmu + subs_lgmhost

    t_obs = _age_at_z_kern(z_obs, *cosmo_params)
    t_0 = _age_at_z_kern(0.0, *cosmo_params)
    lgt0 = np.log10(t_0)

    n_cens = hosts_logmh_at_z.size
    hosts_halo_id = np.arange(n_cens).astype(int)
    _ZH = np.zeros(n_cens)

    tarr = np.zeros(1) + 10**lgt0
    hosts_diffmah = mc_cenpop(
        diffmahpop_params, tarr, hosts_logmh_at_z, t_obs + _ZH, host_key2, lgt0
    )[0]

    n_sats = subs_logmh_at_z.size
    _ZS = np.zeros(n_sats)
    halo_ids = np.arange(n_cens + n_sats).astype(int)
    subs_diffmah = mc_satpop(
        diffmahpop_params, tarr, subs_logmh_at_z, t_obs + _ZS, sub_key2, lgt0
    )[0]

    # For every sub, get diffmah params of its host halo
    subs_host_diffmah = DiffmahParams(*[x[subs_host_halo_indx] for x in hosts_diffmah])

    mah_params = DiffmahParams(
        *[np.concatenate((x, y)) for x, y in zip(hosts_diffmah, subs_diffmah)]
    )
    logmp0 = np.array(_log_mah_kern(mah_params, t_0, lgt0))
    lgmp_t_obs = np.array(_log_mah_kern(mah_params, t_obs, lgt0))

    host_mah_params = DiffmahParams(
        *[np.concatenate((x, y)) for x, y in zip(hosts_diffmah, subs_host_diffmah)]
    )

    subs_lgmp_pen_inf = _log_mah_kern(subs_diffmah, subs_diffmah.t_peak, lgt0)
    subs_lgmp_ult_inf = subs_lgmp_pen_inf

    subs_lgmhost_pen_inf = _log_mah_kern(subs_host_diffmah, subs_diffmah.t_peak, lgt0)
    subs_lgmhost_ult_inf = subs_lgmhost_pen_inf

    lgmp_pen_inf = np.concatenate((hosts_logmh_at_z, subs_lgmp_pen_inf))
    lgmp_ult_inf = np.concatenate((hosts_logmh_at_z, subs_lgmp_ult_inf))

    lgmhost_pen_inf = np.concatenate((hosts_logmh_at_z, subs_lgmhost_pen_inf))
    lgmhost_ult_inf = np.concatenate((hosts_logmh_at_z, subs_lgmhost_ult_inf))

    t_pen_inf = np.concatenate((hosts_diffmah.t_peak, subs_host_diffmah.t_peak))
    t_ult_inf = np.concatenate((hosts_diffmah.t_peak, subs_host_diffmah.t_peak))

    upids = np.concatenate(
        (np.zeros(n_cens).astype(int) - 1, hosts_halo_id[subs_host_halo_indx])
    )

    pen_host_indx = np.concatenate((np.arange(n_cens).astype(int), subs_host_halo_indx))
    ult_host_indx = np.concatenate((np.arange(n_cens).astype(int), subs_host_halo_indx))
    subcat = SubhaloCatalog(
        halo_ids,
        mah_params,
        host_mah_params,
        logmp0,
        lgmp_t_obs,
        lgmp_pen_inf,
        lgmp_ult_inf,
        lgmhost_pen_inf,
        lgmhost_ult_inf,
        t_obs,
        t_pen_inf,
        t_ult_inf,
        upids,
        pen_host_indx,
        ult_host_indx,
    )
    return subcat


def mc_host_halos(
    ran_key,
    z_obs,
    lgmp_min=None,
    volume_com=None,
    hosts_logmh_at_z=None,
    cosmo_params=DEFAULT_COSMOLOGY,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    logmp_cutoff=0.0,
):
    """Monte Carlo realization of a subhalo catalog at a single redshift"""
    host_key1, host_key2 = jran.split(ran_key, 2)
    if hosts_logmh_at_z is None:
        msg = "Must pass volume_com argument if not passing hosts_logmh_at_z"
        assert volume_com is not None, msg
        msg = "Must pass lgmp_min argument if not passing hosts_logmh_at_z"
        assert lgmp_min is not None, msg

        hosts_logmh_at_z = mc_host_halos_singlez(host_key1, lgmp_min, z_obs, volume_com)

    hosts_logmh_at_z_clipped = np.clip(hosts_logmh_at_z, logmp_cutoff, np.inf)

    t_obs = _age_at_z_kern(z_obs, *cosmo_params)
    t_0 = age_at_z0(*cosmo_params)
    lgt0 = np.log10(t_0)

    n_cens = hosts_logmh_at_z.size
    halo_ids = np.arange(n_cens).astype(int)
    _ZH = np.zeros(n_cens)

    tarr = np.zeros(1) + 10**lgt0
    mah_params_uncorrected = mc_cenpop(
        diffmahpop_params, tarr, hosts_logmh_at_z_clipped, t_obs + _ZH, host_key2, lgt0
    )[0]
    lgmp_t_obs_orig = _log_mah_kern(mah_params_uncorrected, t_obs, lgt0)
    delta_logmh_clip = lgmp_t_obs_orig - hosts_logmh_at_z

    mah_params = mah_params_uncorrected._replace(
        logm0=mah_params_uncorrected.logm0 - delta_logmh_clip
    )

    host_mah_params = mah_params
    lgmhost_pen_inf = np.copy(hosts_logmh_at_z)
    lgmhost_ult_inf = np.copy(hosts_logmh_at_z)
    t_pen_inf = mah_params.t_peak
    t_ult_inf = mah_params.t_peak
    upids = _ZH - 1
    pen_host_indx = np.arange(n_cens)
    ult_host_indx = np.arange(n_cens)

    logmp0 = _log_mah_kern(mah_params, 10**lgt0, lgt0)
    lgmp_t_obs = _log_mah_kern(mah_params, t_obs, lgt0)
    lgmp_pen_inf = _log_mah_kern(mah_params, mah_params.t_peak, lgt0)
    lgmp_ult_inf = _log_mah_kern(mah_params, mah_params.t_peak, lgt0)

    subcat = SubhaloCatalog(
        halo_ids,
        mah_params,
        host_mah_params,
        logmp0,
        lgmp_t_obs,
        lgmp_pen_inf,
        lgmp_ult_inf,
        lgmhost_pen_inf,
        lgmhost_ult_inf,
        t_obs,
        t_pen_inf,
        t_ult_inf,
        upids,
        pen_host_indx,
        ult_host_indx,
    )
    return subcat
