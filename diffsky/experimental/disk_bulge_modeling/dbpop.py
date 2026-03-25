""""""

from diffstar.utils import cumulative_mstar_formed_galpop
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ...utils import _sigmoid
from . import disk_bulge_kernels as dbk

FDD_MIN = 0.1
FDD_MAX = 0.9

TCRIT_FRAC = 0.25
FBULGE_EARLY_DD = 0.3
FBULGE_LATE_DD = 0.1
FBULGE_EARLY_BD = 0.9
FBULGE_LATE_BD = 0.7


interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))
interp_vmap2 = jjit(vmap(jnp.interp, in_axes=(0, 0, None)))


@jjit
def _frac_disk_dom_kern(logsm, logssfr):
    delta_fdd = _sigmoid(logssfr, -10.5, 2.0, -0.25, 0.25)
    ylo = jnp.clip(0.9 + delta_fdd, min=FDD_MIN, max=FDD_MAX)
    yhi = jnp.clip(0.1 + delta_fdd, min=FDD_MIN, max=FDD_MAX)
    return _sigmoid(logsm, 10.75, 1.5, ylo, yhi)


@jjit
def mc_fbulge_params(ran_key, tarr, sfh_pop, t_obs):
    fbulge_tcrit, logsm_obs, logssfr_obs = get_fbulge_tcrit(tarr, sfh_pop, t_obs)
    fbulge_early, fbulge_late = mc_fbulge_early_late(ran_key, logsm_obs, logssfr_obs)
    fbulge_params = dbk.DEFAULT_FBULGE_PARAMS._make(
        (fbulge_tcrit, fbulge_early, fbulge_late)
    )
    return fbulge_params


@jjit
def mc_fbulge_early_late(ran_key, logsm, logssfr):
    uran = jran.uniform(ran_key, shape=logsm.shape)
    fdd = _frac_disk_dom_kern(logsm, logssfr)
    fbulge_early = jnp.where(uran < fdd, FBULGE_EARLY_DD, FBULGE_EARLY_BD)
    fbulge_late = jnp.where(uran < fdd, FBULGE_LATE_DD, FBULGE_LATE_BD)

    return fbulge_early, fbulge_late


@jjit
def get_fbulge_tcrit(tarr, sfh_pop, t_obs):
    smh_pop = cumulative_mstar_formed_galpop(tarr, sfh_pop)
    logsmh_pop = jnp.log10(smh_pop)
    logsm_obs = interp_vmap(t_obs, tarr, logsmh_pop)
    sfr_obs = interp_vmap(t_obs, tarr, sfh_pop)
    ssfr_obs = sfr_obs / 10**logsm_obs
    logssfr_obs = jnp.log10(ssfr_obs)

    logsmh_pop_clipped = jnp.clip(logsmh_pop, max=logsm_obs.reshape((-1, 1)))
    fbulge_tcrit = dbk.calc_tform_pop(tarr, 10**logsmh_pop_clipped, 0.25)
    return fbulge_tcrit, logsm_obs, logssfr_obs
