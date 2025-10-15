""""""

import numpy as np
from diffstar.diffstarpop import DEFAULT_DIFFSTARPOP_PARAMS, mc_diffstar_sfh_galpop
from diffstar.utils import cumulative_mstar_formed_galpop
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ...utils import _sigmoid
from ..mc_lightcone_halos import mc_weighted_halo_lightcone

try:
    from halotools.utils import sliding_conditional_percentile

    HAS_HALOTOOLS = True
except ImportError:
    HAS_HALOTOOLS = False


interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))


def compute_ssfr_percentile_tables(
    z_obs, diffstarpop_params=DEFAULT_DIFFSTARPOP_PARAMS, seed=0
):
    if not HAS_HALOTOOLS:
        raise ImportError("Must have halotools installed to call this function")

    ran_key = jran.key(seed)

    num_halos = 5_000
    z_min, z_max = z_obs.min(), z_obs.max()
    lgmp_min, lgmp_max = 8, 15.5
    sky_area_degsq = 10.0

    ran_key, lc_key = jran.split(ran_key, 2)
    args = (lc_key, num_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)

    cenpop = mc_weighted_halo_lightcone(*args)

    ran_key, sfh_table_key = jran.split(ran_key, 2)

    upids = np.zeros_like(cenpop["logmp0"]).astype(int) - 1
    lgmu_infall = np.copy(cenpop["logmp0"])
    logmhost_infall = np.copy(cenpop["logmp0"])
    gyr_since_infall = np.zeros_like(cenpop["logmp0"])

    t_table = np.linspace(0.1, 13.8, 100)

    args = (
        diffstarpop_params,
        cenpop["mah_params"],
        cenpop["logmp0"],
        upids,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        sfh_table_key,
        t_table,
    )

    _res = mc_diffstar_sfh_galpop(*args)
    sfh_params_ms, sfh_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res
    sfh = np.where(mc_is_q.reshape((-1, 1)), sfh_q, sfh_ms)

    smh = cumulative_mstar_formed_galpop(t_table, sfh)

    sfr_obs = interp_vmap(cenpop["t_obs"], t_table, sfh)
    logsm_obs = interp_vmap(cenpop["t_obs"], t_table, np.log10(smh))
    logssfr_obs = np.log10(sfr_obs) - logsm_obs

    ssfr_percentile = sliding_conditional_percentile(logsm_obs, logssfr_obs, 101)
    return logsm_obs, logssfr_obs, ssfr_percentile


def approximate_ssfr_percentile(ssfr, eps=1e-4):
    p = _sigmoid(ssfr, -10.0, 2.0, 0.0, 1.0)
    p = np.clip(p, eps, 1 - eps)
    return p
