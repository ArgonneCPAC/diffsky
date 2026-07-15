# diffsky/lightcone_generators/lc_sfh_centrals.py

"""Lightcone generators of central galaxies with diffstar SFH."""

from collections import namedtuple

from diffstar import calc_sfh_galpop
from diffstar.defaults import FB
from diffstar.diffstarpop import mc_diffstar_sfh_galpop
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from diffstar.diffstarpop.param_utils import mc_select_diffstar_params
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import jit as jjit
from jax import numpy as jnp
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

SFHLightcone = namedtuple("SFHLightcone", _sfh_fields)


def mc_lc_sfh(
    ran_key,
    lc_data,
    *,
    diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS,
    cosmo_params=DEFAULT_COSMOLOGY,
    fb=FB,
):
    mah_params = lc_data.mah_params
    logmp0 = lc_data.logmp0
    lgt0 = lc_data.lgt0
    t_table = lc_data.t_table
    t_obs = lc_data.t_obs

    n_tot = lc_data.z_obs.size
    upid = -jnp.ones(n_tot).astype(int)
    lgmu_infall = jnp.zeros(n_tot)
    logmhost_infall = lc_data.logmp_obs
    gyr_since_infall = -jnp.ones(n_tot)

    args = (
        diffstarpop_params,
        mah_params,
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
        sfh_params, mah_params, t_table, lgt0=lgt0, fb=fb, return_smh=True
    )
    logsm_obs = vmap_interp(t_obs, t_table, jnp.log10(smh_mc))
    logsfr_obs = vmap_interp(t_obs, t_table, jnp.log10(sfh_mc))
    logssfr_obs = logsfr_obs - logsm_obs

    sfh_lightcone = SFHLightcone(
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
