"""The mc_infall_time function is a Monte Carlo generator of subhalo infall times"""

from jax import jit as jjit
from scipy.stats import argus

from ..utils import _sigmoid
from .utils import get_1d_arrays

ARGUS_CHI = 1.5
ARGUS_DT_X0 = -1.25
ARGUS_DT_K = 2.0

DT_LGMU_T0 = 9.0  # Gyr
DT_LGMU_K = 1 / 3.0

DT_LO_LGMU_LO, DT_LO_LGMU_HI = 0.65, 0.4
DT_HI_LGMU_LO, DT_HI_LGMU_HI = 0.875, 0.45


def mc_time_since_infall(lgmu, t_obs, random_state=None):
    """Monte Carlo generator of subhalo t_infall

    Parameters
    ----------
    lgmu : float or ndarray of shape (n, )
        Base-10 log of μ = Msub/Mhost

    t_obs : float or ndarray of shape (n, )
        Age of the universe in Gyr at the time of observation

    Returns
    -------
    time_since_infall : float or ndarray of shape (n, )
        Infall time of the subhalo(s)

    """
    lgmu, t_obs = get_1d_arrays(lgmu, t_obs)

    u_infall = 1 - argus.rvs(ARGUS_CHI, size=lgmu.size, random_state=random_state)
    dt_argus = _get_dt_argus(lgmu, t_obs)
    dimless_infall_time = u_infall * dt_argus
    time_since_infall = dimless_infall_time * t_obs

    if time_since_infall.shape == (1,):
        time_since_infall = time_since_infall[0]

    return time_since_infall


@jjit
def _get_dt_lgmu_lo(lgmu):
    return _sigmoid(lgmu, ARGUS_DT_X0, ARGUS_DT_K, DT_LO_LGMU_LO, DT_LO_LGMU_HI)


@jjit
def _get_dt_lgmu_hi(lgmu):
    return _sigmoid(lgmu, ARGUS_DT_X0, ARGUS_DT_K, DT_HI_LGMU_LO, DT_HI_LGMU_HI)


@jjit
def _get_dt_argus(lgmu, t_obs):
    dt_lo = _get_dt_lgmu_lo(lgmu)
    dt_hi = _get_dt_lgmu_hi(lgmu)
    dt_argus = _sigmoid(t_obs, DT_LGMU_T0, DT_LGMU_K, dt_lo, dt_hi)
    return dt_argus
