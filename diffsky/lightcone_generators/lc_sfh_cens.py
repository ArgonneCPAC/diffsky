"""Lightcone generators of central galaxies with diffstar SFH"""

from collections import namedtuple

from diffhalos.lightcone_generators import mc_lightcone_halos as mclh
from diffmah import logmh_at_t_obs
from diffstar import calc_sfh_galpop
from diffstar.defaults import FB
from diffstar.diffstarpop import mc_diffstar_sfh_galpop
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstar.diffstarpop.param_utils import mc_select_diffstar_params
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import DEFAULT_COSMOLOGY, flat_wcdm
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

N_SFH_TABLE = 100

vmap_interp = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))

_sfh_fields = [
    "t_table",
    "sfh_table",
    "diffstar_params",
    "logsm_obs",
    "logssfr_obs",
    "diffstar_params_ms",
    "diffstar_params_q",
    "sfh_ms",
    "sfh_q",
    "frac_q",
    "mc_is_q",
]
_fields = list(mclh.CenPop._fields) + _sfh_fields
SFHLightcone = namedtuple("SFHLightcone", _fields)


def weighted_lc_halos_sfh(
    ran_key,
    n_host_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    *,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    """Weighted lightcone of host halos with diffstar SFH

    Parameters
    ----------
    ran_key : jax.random.key()
    n_host_halos : int
    z_min, z_max : floats
    lgmp_min, lgmp_max : floats
    sky_area_degsq : float

    Returns
    -------
    sfh_lightcone : namedtuple

    """
    ran_key, halo_key = jran.split(ran_key, 2)
    args = (halo_key, n_host_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)
    halopop = mclh.weighted_lc_halos(*args, cosmo_params=cosmo_params)

    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t0)

    n_tot = halopop.z_obs.size
    upid = -jnp.ones(n_tot).astype(int)
    lgmu_infall = jnp.zeros(n_tot)
    logmhost_infall = halopop.logmp_obs
    gyr_since_infall = -jnp.ones(n_tot)

    logmp0 = logmh_at_t_obs(halopop.mah_params, jnp.zeros(n_tot) + t0, lgt0)
    t_table = jnp.linspace(T_TABLE_MIN, t0, N_SFH_TABLE)

    args = (
        diffstarpop_params,
        halopop.mah_params,
        logmp0,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
    )
    sfh_results = mc_diffstar_sfh_galpop(*args, lgt0=lgt0, fb=fb)

    sfh_params = mc_select_diffstar_params(
        sfh_results.diffstar_params_q,
        sfh_results.diffstar_params_ms,
        sfh_results.mc_is_q,
    )
    sfh_mc, smh_mc = calc_sfh_galpop(
        sfh_params, halopop.mah_params, t_table, lgt0=lgt0, fb=fb, return_smh=True
    )
    logsm_obs = vmap_interp(halopop.t_obs, t_table, jnp.log10(smh_mc))
    logsfr_obs = vmap_interp(halopop.t_obs, t_table, jnp.log10(sfh_mc))
    logssfr_obs = logsfr_obs - logsm_obs

    sfh_lightcone = SFHLightcone(
        *halopop,
        t_table,
        sfh_mc,
        sfh_params,
        logsm_obs,
        logssfr_obs,
        sfh_results.diffstar_params_ms,
        sfh_results.diffstar_params_q,
        sfh_results.sfh_ms,
        sfh_results.sfh_q,
        sfh_results.frac_q,
        sfh_results.mc_is_q,
    )

    return sfh_lightcone
