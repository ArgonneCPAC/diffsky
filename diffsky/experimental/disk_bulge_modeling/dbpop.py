"""Population-level model of disk/bulge composition"""

from collections import namedtuple

from diffstar.utils import cumulative_mstar_formed_galpop
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ...utils import _sigmoid
from . import disk_bulge_kernels as dbk

FDD_MIN = 0.0
FDD_MAX = 0.9

TCRIT_FRAC = 0.25
FBULGE_EARLY_DD = 0.3
FBULGE_LATE_DD = 0.1
FBULGE_EARLY_BD = 0.9
FBULGE_LATE_BD = 0.7


interp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))
interp_vmap2 = jjit(vmap(jnp.interp, in_axes=(0, 0, None)))


_DB = (
    "fbulge_params",
    "mstar_history",
    "eff_bulge_history",
    "sfh_bulge",
    "smh_bulge",
    "bulge_to_total_history",
)
DiskBulgeHistory = namedtuple("DiskBulgeSFH", _DB)


def decompose_sfh_into_disk_bulge_sfh(uran_fbulge, tarr, sfh_pop, t_obs):
    """Calculate disk/bulge decomposition of composite galaxy SFH

    Parameters
    ----------
    uran_bulge : array, shape (n_gals, )
        Randoms used to stochastically assign galaxy to be disk- or bulge-dominated

    tarr : array, shape (n_t, )
        Array of cosmic time in Gyr

    sfh_pop : array, shape (n_gals, n_t)
        Star formation history in Msun/yr

    t_obs : array, shape (n_gals, )
        Age of the universe at the time of observation in Gyr

    """
    fbulge_params = get_fbulge_params(uran_fbulge, tarr, sfh_pop, t_obs)
    _res = dbk._bulge_sfh_vmap(tarr, sfh_pop, fbulge_params)
    smh, eff_bulge, sfh_bulge, smh_bulge, bth = _res
    return DiskBulgeHistory(fbulge_params, smh, eff_bulge, sfh_bulge, smh_bulge, bth)


@jjit
def _frac_disk_dom_kern(logsm, logssfr):
    """Fraction of disk-dominated galaxies"""
    delta_fdd = _sigmoid(logssfr, -10.5, 2.0, -0.25, 0.25)
    ylo = jnp.clip(0.9 + delta_fdd, min=FDD_MIN, max=FDD_MAX)
    yhi = jnp.clip(0.0, min=FDD_MIN, max=FDD_MAX)
    return _sigmoid(logsm, 10.25, 2, ylo, yhi)


@jjit
def get_fbulge_params(uran_fbulge, tarr, sfh_pop, t_obs):
    """Calculate parameters that determine how SFH is apportioned into disk/bulge"""
    fbulge_tcrit, logsm_obs, logssfr_obs = get_fbulge_tcrit(tarr, sfh_pop, t_obs)
    fbulge_early, fbulge_late = get_fbulge_early_late(
        uran_fbulge, logsm_obs, logssfr_obs
    )
    fbulge_params = dbk.DEFAULT_FBULGE_PARAMS._make(
        (fbulge_tcrit, fbulge_early, fbulge_late)
    )
    return fbulge_params


@jjit
def get_fbulge_early_late(uran_fbulge, logsm, logssfr):
    """Calculate sigmoid params that determine time-evolution of bulge formation efficiency

    fbulge_early > fbulge_late ==> bulges are redder than disks within the same galaxy

    """
    fdd = _frac_disk_dom_kern(logsm, logssfr)
    fbulge_early = jnp.where(uran_fbulge < fdd, FBULGE_EARLY_DD, FBULGE_EARLY_BD)
    fbulge_late = jnp.where(uran_fbulge < fdd, FBULGE_LATE_DD, FBULGE_LATE_BD)

    return fbulge_early, fbulge_late


@jjit
def get_fbulge_tcrit(tarr, sfh_pop, t_obs):
    """Calculate characteristic time bulge efficiency drops"""
    smh_pop = cumulative_mstar_formed_galpop(tarr, sfh_pop)
    logsmh_pop = jnp.log10(smh_pop)
    logsm_obs = interp_vmap(t_obs, tarr, logsmh_pop)
    sfr_obs = interp_vmap(t_obs, tarr, sfh_pop)
    ssfr_obs = sfr_obs / 10**logsm_obs
    logssfr_obs = jnp.log10(ssfr_obs)

    logsmh_pop_clipped = jnp.clip(logsmh_pop, max=logsm_obs.reshape((-1, 1)))
    fbulge_tcrit = dbk.calc_tform_pop(tarr, 10**logsmh_pop_clipped, 0.25)
    return fbulge_tcrit, logsm_obs, logssfr_obs
