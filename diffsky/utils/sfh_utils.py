""" """

from diffstar.utils import cumulative_mstar_formed
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap


@jjit
def _logsm_logssfr_at_t_obs_kern(t_obs, t_table, sfr_table):
    lgt_obs = jnp.log10(t_obs)

    lgt_table = jnp.log10(t_table)
    mstar_table = cumulative_mstar_formed(t_table, sfr_table)
    logsm_table = jnp.log10(mstar_table)
    logsfr_table = jnp.log10(sfr_table)
    logsm_obs = jnp.interp(lgt_obs, lgt_table, logsm_table)
    logsfr_obs = jnp.interp(lgt_obs, lgt_table, logsfr_table)
    logssfr_obs = logsfr_obs - logsm_obs
    return logsm_obs, logssfr_obs


_A = (0, None, 0)
_logsm_logssfr_at_t_obs_vmap = jjit(vmap(_logsm_logssfr_at_t_obs_kern, in_axes=_A))


@jjit
def get_logsm_logssfr_at_t_obs(t_obs, t_table, sfh):
    """Compute log10Mstar(t_obs) and log10sSFR(t_obs) for a galaxy population

    Parameters
    ----------
    t_obs : array, shape (n_gals, )

    t_table : array, shape (n_table, )

    sfh : array, shape (n_gals, n_table)

    Returns
    -------
    logsm_obs : array, shape (n_gals, )

    logssfr_obs : array, shape (n_gals, )

    """
    logsm_obs, logssfr_obs = _logsm_logssfr_at_t_obs_vmap(t_obs, t_table, sfh)
    return logsm_obs, logssfr_obs
