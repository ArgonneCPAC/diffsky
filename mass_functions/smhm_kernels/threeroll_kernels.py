"""Functions predicting Mstar and the SMF from the SMHM params."""

from collections import OrderedDict

from jax import jit as jjit
from jax import numpy as jnp

LOG10MSTAR = 10.85

DEFAULT_SMHM_PARAMS = OrderedDict(
    smhm_logm_crit=11.35,
    smhm_ratio_logm_crit=-1.65,
    smhm_k_logm=1.6,
    smhm_lowm_index_x0=11.5,
    smhm_lowm_index_k=2,
    smhm_lowm_index_ylo=2.5,
    smhm_lowm_index_yhi=2.5,
    smhm_highm_index_x0=13.5,
    smhm_highm_index_k=2,
    smhm_highm_index_ylo=0.5,
    smhm_highm_index_yhi=0.5,
)
DEFAULT_SMHM_PARAM_BOUNDS = OrderedDict(
    smhm_logm_crit=(9.0, 16.0),
    smhm_ratio_logm_crit=(-5.0, 0.0),
    smhm_k_logm=(0.0, 25.0),
    smhm_lowm_index_x0=(9.0, 16.0),
    smhm_lowm_index_k=(0.0, 25.0),
    smhm_lowm_index_ylo=(0.0, 10.0),
    smhm_lowm_index_yhi=(0.0, 10.0),
    smhm_highm_index_x0=(9.0, 16.5),
    smhm_highm_index_k=(0.0, 15.0),
    smhm_highm_index_ylo=(-0.5, 15.0),
    smhm_highm_index_yhi=(-0.5, 15.0),
)

_PBOUND_X0, _PBOUND_K = 0.0, 0.1
BOUNDS = jnp.array(list(DEFAULT_SMHM_PARAM_BOUNDS.values()))
SCATTER_INFLECTION, SCATTER_K = 12.0, 1.0


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def _bounded_params_from_unbounded(up):
    p = [_sigmoid(up[i], _PBOUND_X0, _PBOUND_K, *BOUNDS[i]) for i in range(len(up))]
    return jnp.array(p)


@jjit
def _unbounded_params_from_bounded(p):
    up = [
        _inverse_sigmoid(p[i], _PBOUND_X0, _PBOUND_K, *BOUNDS[i]) for i in range(len(p))
    ]
    return jnp.array(up)


U_PARAMS = _unbounded_params_from_bounded(list(DEFAULT_SMHM_PARAMS.values()))
DEFAULT_SMHM_U_PARAMS = OrderedDict(
    [(s, v) for s, v in zip(DEFAULT_SMHM_PARAMS.keys(), U_PARAMS)]
)


@jjit
def _logsm_from_logmh(smhm_params, logmh):
    """Kernel of the three-roll function mapping Mhalo ==> Mstar.

    Parameters
    ----------
    smhm_params : ndarray, shape (11, )
        Parameters of the three-roll function used to map Mhalo ==> Mstar,

    logmh : ndarray, shape (n, )
        Base-10 log of halo mass

    Returns
    -------
    logsm : ndarray, shape (n, )
        Base-10 log of stellar mass

    """
    logm_crit, log_sfeff_at_logm_crit, smhm_k_logm = smhm_params[0:3]
    lo_indx_pars = smhm_params[3:7]
    hi_indx_pars = smhm_params[7:11]

    lowm_index = _sigmoid(logmh, *lo_indx_pars)
    highm_index = _sigmoid(logmh, *hi_indx_pars)

    logsm_at_logm_crit = logm_crit + log_sfeff_at_logm_crit
    powerlaw_index = _sigmoid(logmh, logm_crit, smhm_k_logm, lowm_index, highm_index)

    return logsm_at_logm_crit + powerlaw_index * (logmh - logm_crit)
