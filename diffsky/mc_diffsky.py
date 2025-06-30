""" """

import os

from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffstar.defaults import T_TABLE_MIN
from diffstar.utils import cumulative_mstar_formed_galpop
from diffstarpop import mc_diffstar_sfh_galpop
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstarpop.param_utils import mc_select_diffstar_params
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .mass_functions.mc_diffmah_tpeak import mc_host_halos, mc_subhalos

try:
    DSPS_DATA_DRN = os.environ["DSPS_DRN"]
except KeyError:
    DSPS_DATA_DRN = ""


N_T = 100

_interp_vmap_single_t_obs = jjit(vmap(jnp.interp, in_axes=(None, None, 0)))


def mc_diffstar_galpop(
    ran_key,
    z_obs,
    lgmp_min,
    volume_com=None,
    hosts_logmh_at_z=None,
    cosmo_params=DEFAULT_COSMOLOGY,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    n_t=N_T,
    return_internal_quantities=False,
):
    """Generate a population of galaxies with diffmah MAH and diffstar SFH

    Parameters
    ----------
    ran_key : jran.PRNGKey

    z_obs : float
        Redshift of the halo population

    lgmp_min : float
        Base-10 log of the halo mass competeness limit of the generated population
        Smaller values of lgmp_min produce more halos in the returned sample
        A small fraction of halos will have slightly smaller masses than lgmp_min

    volume_com : float, optional
        volume_com = Lbox**3 where Lbox is in comoving in units of Mpc/h
        Default is None, in which case argument hosts_logmh_at_z must be passed

        Larger values of volume_com produce more halos in the returned sample

    hosts_logmh_at_z : ndarray, optional
        Grid of host halo masses at the input redshift.
        Default is None, in which case volume_com argument must be passed
        and the host halo mass function will be randomly sampled.

    return_internal_quantities : bool, optional
        If True, returned data will include additional info such as
        the separate SFHs for the probabilistic main and quenched sequences.
        Default is False, in which case only a single SFH will be returned.

    Returns
    -------
    diffsky_data : dict
        Diffstar galaxy population

    """
    mah_key, sfh_key = jran.split(ran_key, 2)

    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    t_table = jnp.linspace(T_TABLE_MIN, t0, n_t)

    subcat = mc_subhalos(
        mah_key,
        z_obs,
        lgmp_min,
        volume_com=volume_com,
        hosts_logmh_at_z=hosts_logmh_at_z,
        cosmo_params=cosmo_params,
        diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    )

    logmu_infall = subcat.logmp_ult_inf - subcat.logmhost_ult_inf
    t_obs = flat_wcdm._age_at_z_kern(z_obs, *cosmo_params)
    args = (
        diffstarpop_params,
        subcat.mah_params,
        subcat.logmp0,
        subcat.upids,
        logmu_infall,
        subcat.logmhost_ult_inf,
        t_obs - subcat.t_ult_inf,
        sfh_key,
        t_table,
    )

    _res = mc_diffstar_sfh_galpop(*args)
    sfh_ms, sfh_q, frac_q, mc_is_q = _res[2:]
    sfh_table = jnp.where(mc_is_q.reshape((-1, 1)), sfh_q, sfh_ms)
    smh_table = cumulative_mstar_formed_galpop(t_table, sfh_table)
    diffstar_params_ms, diffstar_params_q = _res[0:2]
    sfh_params = mc_select_diffstar_params(
        diffstar_params_q, diffstar_params_ms, mc_is_q
    )

    diffstar_data = dict()
    diffstar_data["subcat"] = subcat
    diffstar_data["t_table"] = t_table
    diffstar_data["t_obs"] = t_obs
    diffstar_data["sfh"] = sfh_table
    diffstar_data["smh"] = smh_table
    diffstar_data["mc_quenched"] = mc_is_q
    diffstar_data["sfh_params"] = sfh_params

    diffstar_data["logsm_obs"] = _interp_vmap_single_t_obs(
        t_obs, t_table, jnp.log10(diffstar_data["smh"])
    )
    logsfh_obs = _interp_vmap_single_t_obs(
        t_obs, t_table, jnp.log10(diffstar_data["sfh"])
    )
    diffstar_data["logssfr_obs"] = logsfh_obs - diffstar_data["logsm_obs"]

    if return_internal_quantities:
        diffstar_data["sfh_ms"] = sfh_ms
        diffstar_data["sfh_q"] = sfh_q
        diffstar_data["frac_q"] = frac_q
        diffstar_data["sfh_params_ms"] = diffstar_params_ms
        diffstar_data["sfh_params_q"] = diffstar_params_q

    return diffstar_data


def mc_diffstar_cenpop(
    ran_key,
    z_obs,
    lgmp_min=None,
    volume_com=None,
    hosts_logmh_at_z=None,
    cosmo_params=DEFAULT_COSMOLOGY,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    n_t=N_T,
    return_internal_quantities=False,
):
    """Generate a population of central galaxies with diffmah MAH and diffstar SFH

    Parameters
    ----------
    ran_key : jran.PRNGKey

    z_obs : float
        Redshift of the halo population

    lgmp_min : float
        Base-10 log of the halo mass competeness limit of the generated population
        Smaller values of lgmp_min produce more halos in the returned sample
        A small fraction of halos will have slightly smaller masses than lgmp_min

    volume_com : float, optional
        volume_com = Lbox**3 where Lbox is in comoving in units of Mpc/h
        Default is None, in which case argument hosts_logmh_at_z must be passed

        Larger values of volume_com produce more halos in the returned sample

    hosts_logmh_at_z : ndarray, optional
        Grid of host halo masses at the input redshift.
        Default is None, in which case volume_com argument must be passed
        and the host halo mass function will be randomly sampled.

    return_internal_quantities : bool, optional
        If True, returned data will include additional info such as
        the separate SFHs for the probabilistic main and quenched sequences.
        Default is False, in which case only a single SFH will be returned.

    Returns
    -------
    diffsky_data : dict
        Diffstar galaxy population

    """

    mah_key, sfh_key = jran.split(ran_key, 2)

    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    t_table = jnp.linspace(T_TABLE_MIN, t0, n_t)

    subcat = mc_host_halos(
        mah_key,
        z_obs,
        lgmp_min=lgmp_min,
        volume_com=volume_com,
        hosts_logmh_at_z=hosts_logmh_at_z,
        cosmo_params=cosmo_params,
        diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    )

    logmu_infall = subcat.logmp_ult_inf - subcat.logmhost_ult_inf
    t_obs = flat_wcdm._age_at_z_kern(z_obs, *cosmo_params)
    args = (
        diffstarpop_params,
        subcat.mah_params,
        subcat.logmp0,
        subcat.upids,
        logmu_infall,
        subcat.logmhost_ult_inf,
        t_obs - subcat.t_ult_inf,
        sfh_key,
        t_table,
    )

    _res = mc_diffstar_sfh_galpop(*args)
    sfh_ms, sfh_q, frac_q, mc_is_q = _res[2:]
    sfh_table = jnp.where(mc_is_q.reshape((-1, 1)), sfh_q, sfh_ms)
    smh_table = cumulative_mstar_formed_galpop(t_table, sfh_table)
    diffstar_params_ms, diffstar_params_q = _res[0:2]
    sfh_params = mc_select_diffstar_params(
        diffstar_params_q, diffstar_params_ms, mc_is_q
    )

    diffstar_data = dict()
    diffstar_data["subcat"] = subcat
    diffstar_data["t_table"] = t_table
    diffstar_data["t_obs"] = t_obs
    diffstar_data["sfh"] = sfh_table
    diffstar_data["smh"] = smh_table
    diffstar_data["mc_quenched"] = mc_is_q
    diffstar_data["sfh_params"] = sfh_params

    diffstar_data["logsm_obs"] = _interp_vmap_single_t_obs(
        t_obs, t_table, jnp.log10(diffstar_data["smh"])
    )
    logsfh_obs = _interp_vmap_single_t_obs(
        t_obs, t_table, jnp.log10(diffstar_data["sfh"])
    )
    diffstar_data["logssfr_obs"] = logsfh_obs - diffstar_data["logsm_obs"]

    if return_internal_quantities:
        diffstar_data["sfh_ms"] = sfh_ms
        diffstar_data["sfh_q"] = sfh_q
        diffstar_data["frac_q"] = frac_q
        diffstar_data["sfh_params_ms"] = diffstar_params_ms
        diffstar_data["sfh_params_q"] = diffstar_params_q

    return diffstar_data
