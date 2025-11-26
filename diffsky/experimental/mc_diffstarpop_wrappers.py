""""""

from collections import namedtuple

from diffstar.diffstarpop import mc_diffstar_sfh_galpop
from diffstar.utils import cumulative_mstar_formed_galpop
from dsps.cosmology import flat_wcdm
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))


_DPKEYS = (
    "frac_q",
    "sfh_ms",
    "logsm_obs_ms",
    "logssfr_obs_ms",
    "sfh_q",
    "logsm_obs_q",
    "logssfr_obs_q",
    "diffstar_params_ms",
    "diffstar_params_q",
)
DiffstarPopQuantities = namedtuple("DiffstarPopQuantities", _DPKEYS)


@jjit
def _get_sfh_info_at_t_obs(t_table, sfh_table, t_obs):
    logsmh_table = jnp.log10(cumulative_mstar_formed_galpop(t_table, sfh_table))
    logsm_obs = interp_vmap(t_obs, t_table, logsmh_table)
    logsfr_obs = interp_vmap(t_obs, t_table, jnp.log10(sfh_table))
    logssfr_obs = logsfr_obs - logsm_obs
    return logsm_obs, logssfr_obs


@jjit
def diffstarpop_lc_cen_wrapper(
    diffstarpop_params, ran_key, mah_params, logmp0, t_table, t_obs, cosmo_params, fb
):
    n_gals = logmp0.size
    upids = jnp.zeros(n_gals).astype(int) - 1
    lgmu_infall = jnp.zeros(n_gals) - 1.0
    logmhost_infall = jnp.copy(logmp0)
    lgmu_infall = jnp.zeros(n_gals) - 1.0
    gyr_since_infall = jnp.zeros(n_gals)
    lgt0 = jnp.log10(flat_wcdm.age_at_z0(*cosmo_params))

    args = (
        diffstarpop_params,
        mah_params,
        logmp0,
        upids,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
    )
    _res = mc_diffstar_sfh_galpop(*args, lgt0=lgt0, fb=fb)
    diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

    logsm_obs_ms, logssfr_obs_ms = _get_sfh_info_at_t_obs(t_table, sfh_ms, t_obs)
    logsm_obs_q, logssfr_obs_q = _get_sfh_info_at_t_obs(t_table, sfh_q, t_obs)

    diffstar_galpop = DiffstarPopQuantities(
        frac_q=frac_q,
        sfh_ms=sfh_ms,
        logsm_obs_ms=logsm_obs_ms,
        logssfr_obs_ms=logssfr_obs_ms,
        sfh_q=sfh_q,
        logsm_obs_q=logsm_obs_q,
        logssfr_obs_q=logssfr_obs_q,
        diffstar_params_ms=diffstar_params_ms,
        diffstar_params_q=diffstar_params_q,
    )

    return diffstar_galpop
