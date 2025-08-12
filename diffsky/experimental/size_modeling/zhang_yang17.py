"""Module implementing fitting functions for the size-luminosity relation
taken from Zhang & Yang (2017), arXiv:1707.04979.
Used for mdelingf galaxy sizes in the cosmoDC2 and SkySim5000 catalogs
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext

MAX_SIZE = 60.0  # kpc

__all__ = (
    "median_size_vs_luminosity_late_type",
    "mc_size_vs_luminosity_early_type",
    "median_size_vs_luminosity_early_type",
    "mc_size_vs_luminosity_late_type",
)

fixed_seed = 43

#  Disk parameters (see B/T < 0.5 row of Table 1 of 1707.04979)
alpha_disk = 0.32
beta_disk = 1.75
gamma_disk = 12.63
mzero_disk = -24.8
scatter_disk = 0.1

#  Bulge parameters (see B/T > 0.5 row of Table 1 of 1707.04979)
alpha_bulge = 0.33
beta_bulge = 1.0
gamma_bulge = 3.25
mzero_bulge = -22.5
scatter_bulge = 0.1

#  Redshift-dependence parameters
default_z_table = (0.25, 0.75, 1.25, 2)
default_shrinking_table = (1.0, 1.25, 1.5, 2.0)


def redshift_shrinking_factor(redshift, z0=1, ymin=1, ymax=2, k=4):
    """Sigmoid function calibrated against van der Wel 2014."""
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + np.exp(-k * (redshift - z0)))


def median_size_vs_luminosity(magr, redshift, gamma, alpha, beta, mzero):
    """
    Generic functional form used by Zhang & Yang (2017) to model size vs. luminosity

    Parameters
    ----------
    magr : ndarray
        Numpy array of shape (ngals, )
        storing SDSS restframe absolute r-band magnitude

    redshift : ndarray
        Numpy array of shape (ngals, ) storing galaxy redshift

    gamma : float
        normalization parameter of the fitting function

    alpha : float
        faint-end slope parameter of the fitting function

    beta : float
        bright-end slope parameter of the fitting function

    mzero : float
        transition luminosity parameter of the fitting function

    Returns
    -------
    median_size : ndarray
        Numpy array of shape (npts, ) storing the output size in units of kpc

    Notes
    -----
    See Equation (3) of arXiv:1707.04979.

    """
    luminosity = 10 ** (-0.4 * (magr - mzero))
    z0_size = gamma * (luminosity**alpha) * ((1.0 + luminosity) ** (beta - alpha))
    shrinking_factor = redshift_shrinking_factor(redshift)
    size = z0_size / shrinking_factor
    size = np.where(size > MAX_SIZE, MAX_SIZE, size)
    return size


def median_size_vs_luminosity_early_type(
    magr,
    redshift,
    alpha=alpha_bulge,
    beta=beta_bulge,
    gamma=gamma_bulge,
    mzero=mzero_bulge,
):
    """
    Fitting function for median bulge size vs. luminosity

    Parameters
    ----------
    magr : ndarray
        Numpy array of shape (ngals, )
        storing SDSS restframe absolute r-band magnitude

    redshift : ndarray
        Numpy array of shape (ngals, ) storing galaxy redshift

    gamma : float, optional
        normalization parameter of the fitting function

    alpha : float, optional
        faint-end slope parameter of the fitting function

    beta : float, optional
        bright-end slope parameter of the fitting function

    mzero : float, optional
        transition luminosity parameter of the fitting function

    Returns
    -------
    median_size : ndarray
        Numpy array of shape (npts, ) storing the output bulge size in units of kpc

    Notes
    -----
    See Equation (3) of arXiv:1707.04979.

    Examples
    --------
    >>> magr = np.linspace(-25, -10, 500)
    >>> redshift = np.random.uniform(0, 3, 500)
    >>> sizes = median_size_vs_luminosity_early_type(magr, redshift)
    """
    return median_size_vs_luminosity(magr, redshift, gamma, alpha, beta, mzero)


def median_size_vs_luminosity_late_type(
    magr, redshift, alpha=alpha_disk, beta=beta_disk, gamma=gamma_disk, mzero=mzero_disk
):
    """
    Fitting function for median disk size vs. luminosity

    Parameters
    ----------
    magr : ndarray
        Numpy array of shape (ngals, ) storing SDSS restframe absolute r-band magnitude

    redshift : ndarray
        Numpy array of shape (ngals, ) storing galaxy redshift

    gamma : float, optional
        normalization parameter of the fitting function

    alpha : float, optional
        faint-end slope parameter of the fitting function

    beta : float, optional
        bright-end slope parameter of the fitting function

    mzero : float, optional
        transition luminosity parameter of the fitting function

    Returns
    -------
    median_size : ndarray
        Numpy array of shape (npts, ) storing the output disk size in units of kpc

    Notes
    -----
    See Equation (3) of arXiv:1707.04979.

    Examples
    --------
    >>> magr = np.linspace(-25, -10, 500)
    >>> redshift = np.random.uniform(0, 3, 500)
    >>> sizes = median_size_vs_luminosity_late_type(magr, redshift)
    """
    return median_size_vs_luminosity(magr, redshift, gamma, alpha, beta, mzero)


def mc_size_vs_luminosity_early_type(
    magr,
    redshift,
    alpha=alpha_bulge,
    beta=beta_bulge,
    gamma=gamma_bulge,
    mzero=mzero_bulge,
    scatter=scatter_bulge,
    seed=fixed_seed,
):
    """
    Monte Carlo realization of bulge size based on the
    Zhang & Yang (2017) fitting function for median bulge size vs. luminosity

    Parameters
    ----------
    magr : ndarray
        Numpy array of shape (ngals, )
        storing SDSS restframe absolute r-band magnitude

    redshift : ndarray
        Numpy array of shape (ngals, ) storing galaxy redshift

    gamma : float, optional
        normalization parameter of the fitting function

    alpha : float, optional
        faint-end slope parameter of the fitting function

    beta : float, optional
        bright-end slope parameter of the fitting function

    mzero : float, optional
        transition luminosity parameter of the fitting function

    seed : int, optional
        Random number seed in the Monte Carlo. Default is 43.

    Returns
    -------
    median_size : ndarray
        Numpy array of shape (ngals, ) storing the output half-radius of the
        bulge in units of physical kpc assuming h=0.7

    Notes
    -----
    See Equation (3) of arXiv:1707.04979.

    Examples
    --------
    >>> magr = np.linspace(-25, -10, 500)
    >>> redshift = np.random.uniform(0, 3, 500)
    >>> sizes = mc_size_vs_luminosity_early_type(magr, redshift)
    """
    loc = np.log10(median_size_vs_luminosity(magr, redshift, gamma, alpha, beta, mzero))
    with NumpyRNGContext(seed):
        size = 10 ** np.random.normal(loc=loc, scale=scatter)
    size = np.where(size > MAX_SIZE, MAX_SIZE, size)
    return size


def mc_size_vs_luminosity_late_type(
    magr,
    redshift,
    alpha=alpha_disk,
    beta=beta_disk,
    gamma=gamma_disk,
    mzero=mzero_disk,
    scatter=scatter_disk,
    seed=fixed_seed,
):
    """
    Monte Carlo realization of disk size based on the
    Zhang & Yang (2017) fitting function for median disk size vs. luminosity

    Parameters
    ----------
    magr : ndarray
        Numpy array of shape (ngals, ) storing SDSS restframe absolute r-band magnitude

    redshift : ndarray
        Numpy array of shape (ngals, ) storing galaxy redshift

    gamma : float, optional
        normalization parameter of the fitting function

    alpha : float, optional
        faint-end slope parameter of the fitting function

    beta : float, optional
        bright-end slope parameter of the fitting function

    mzero : float, optional
        transition luminosity parameter of the fitting function

    seed : int, optional
        Random number seed in the Monte Carlo. Default is 43.

    Returns
    -------
    median_size : ndarray
        Numpy array of shape (ngals, ) storing the output half-radius of the
        disk in units of physical kpc assuming h=0.7

    Notes
    -----
    See Equation (3) of arXiv:1707.04979.

    Examples
    --------
    >>> magr = np.linspace(-25, -10, 500)
    >>> redshift = np.random.uniform(0, 3, 500)
    >>> sizes = mc_size_vs_luminosity_late_type(magr, redshift)
    """
    loc = np.log10(median_size_vs_luminosity(magr, redshift, gamma, alpha, beta, mzero))
    with NumpyRNGContext(seed):
        size = 10 ** np.random.normal(loc=loc, scale=scatter)
    size = np.where(size > MAX_SIZE, MAX_SIZE, size)
    return size
