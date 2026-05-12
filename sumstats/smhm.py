"""JAX kernels for N-dimensional triweighted histograms"""

from jax import jit as jjit
from jax import numpy as jnp

from .diffndhist import tw_ndhist_weighted


@jjit
def compute_smhm(logmh, logsm, sigma, logmh_bins):
    """Differentiable calculation of the stellar-to-halo mass relation

    Parameters
    ----------
    logmh : array, shape (n, )
        log10 halo mass

    logsm : array, shape (n, )
        log10 stellar mass

    sigma : array, shape (n, )
        logsm scatter

    logmh_bins : array, shape (n_bins, )

    Returns
    ----------
    mean_logsm : array, shape (n_bins, )

    """
    nddata = logmh.reshape((-1, 1))
    ndsig = sigma.reshape((-1, 1))
    ydata = logsm.reshape((-1, 1))
    ndbins_lo = logmh_bins[:-1].reshape((-1, 1))
    ndbins_hi = logmh_bins[1:].reshape((-1, 1))

    wndhist = tw_ndhist_weighted(nddata, ndsig, ydata, ndbins_lo, ndbins_hi)
    _ones = jnp.ones_like(ydata)
    wcounts = tw_ndhist_weighted(nddata, ndsig, _ones, ndbins_lo, ndbins_hi)

    mean_logsm0_tw = wndhist / wcounts

    return mean_logsm0_tw
