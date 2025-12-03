""""""

from collections import namedtuple
from functools import partial

from diffmah import logmh_at_t_obs
from diffstar import calc_sfh_galpop
from diffstar.diffstarpop import mc_diffstar_params_galpop, mc_diffstar_sfh_galpop
from diffstar.diffstarpop.param_utils import mc_select_diffstar_params
from diffstar.utils import cumulative_mstar_formed_galpop
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

MCDiffstarParams = namedtuple(
    "MCDiffstarParams", ("diffstar_params_ms", "diffstar_params_q", "frac_q", "mc_is_q")
)
MCDiffstar = namedtuple(
    "MCDiffstar",
    ("diffstar_params_ms", "diffstar_params_q", "sfh_ms", "sfh_q", "frac_q", "mc_is_q"),
)

interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))

N_T_TABLE = 100

_DPKEYS = (
    "logmp0",
    "logmp_obs",
    "frac_q",
    "sfh_ms",
    "logsm_obs_ms",
    "logssfr_obs_ms",
    "sfh_q",
    "logsm_obs_q",
    "logssfr_obs_q",
    "diffstar_params_ms",
    "diffstar_params_q",
    "t_table",
)
DiffstarPopQuantities = namedtuple("DiffstarPopQuantities", _DPKEYS)


@jjit
def _get_sfh_info_at_t_obs(t_table, sfh_table, t_obs):
    logsmh_table = jnp.log10(cumulative_mstar_formed_galpop(t_table, sfh_table))
    logsm_obs = interp_vmap(t_obs, t_table, logsmh_table)
    logsfr_obs = interp_vmap(t_obs, t_table, jnp.log10(sfh_table))
    logssfr_obs = logsfr_obs - logsm_obs
    return logsm_obs, logssfr_obs


@partial(jjit, static_argnames=["n_t_table"])
def diffstarpop_cen_wrapper(
    diffstarpop_params,
    ran_key,
    mah_params,
    t_obs,
    cosmo_params,
    fb,
    n_t_table=N_T_TABLE,
):
    n_gals = mah_params.logm0.size
    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t0)
    t_table = jnp.linspace(T_TABLE_MIN, t0, n_t_table)
    ZZ = jnp.zeros(n_gals)

    logmp_obs = logmh_at_t_obs(mah_params, t_obs + ZZ, lgt0)
    logmp0 = logmh_at_t_obs(mah_params, t0 + ZZ, lgt0)

    upids = jnp.zeros(n_gals).astype(int) - 1
    lgmu_infall = jnp.zeros(n_gals) - 1.0
    logmhost_infall = jnp.copy(logmp0)
    lgmu_infall = jnp.zeros(n_gals) - 1.0
    gyr_since_infall = jnp.zeros(n_gals)

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
        logmp0=logmp0,
        logmp_obs=logmp_obs,
        frac_q=frac_q,
        sfh_ms=sfh_ms,
        logsm_obs_ms=logsm_obs_ms,
        logssfr_obs_ms=logssfr_obs_ms,
        sfh_q=sfh_q,
        logsm_obs_q=logsm_obs_q,
        logssfr_obs_q=logssfr_obs_q,
        diffstar_params_ms=diffstar_params_ms,
        diffstar_params_q=diffstar_params_q,
        t_table=t_table,
    )

    return diffstar_galpop


@jjit
def mc_diffstarpop_cens_wrapper(diffstarpop_params, ran_key, mah_params, cosmo_params):
    n_gals = mah_params.logm0.size
    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t0)
    ZZ = jnp.zeros(n_gals)

    logmp0 = logmh_at_t_obs(mah_params, t0 + ZZ, lgt0)

    upid = jnp.zeros(n_gals).astype(int) - 1
    lgmu_infall = jnp.zeros(n_gals) - 1.0
    logmhost_infall = jnp.copy(logmp0)
    lgmu_infall = jnp.zeros(n_gals) - 1.0
    gyr_since_infall = jnp.zeros(n_gals)

    _res = mc_diffstar_params_galpop(
        diffstarpop_params,
        logmp0,
        mah_params.t_peak,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    diffstar_params_ms, diffstar_params_q, frac_q, mc_is_q = _res
    sfh_params = mc_select_diffstar_params(
        diffstar_params_q, diffstar_params_ms, mc_is_q
    )
    return sfh_params, mc_is_q


@jjit
def compute_diffstar_sfh_wrapper(sfh_params, mah_params, t_table, lgt0, fb):
    sfh_table = calc_sfh_galpop(sfh_params, mah_params, t_table, lgt0=lgt0, fb=fb)
    return sfh_table


@partial(jjit, static_argnames=["n_t_table"])
def compute_diffstar_info(mah_params, sfh_params, t_obs, cosmo_params, fb, n_t_table):
    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t0)
    t_table = jnp.linspace(T_TABLE_MIN, t0, n_t_table)
    sfh_table = compute_diffstar_sfh_wrapper(sfh_params, mah_params, t_table, lgt0, fb)
    logsm_obs, logssfr_obs = _get_sfh_info_at_t_obs(t_table, sfh_table, t_obs)
    return t_table, sfh_table, logsm_obs, logssfr_obs
