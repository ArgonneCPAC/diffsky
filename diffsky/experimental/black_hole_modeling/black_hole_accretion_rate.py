import numpy as np

A = 10**-3.15
gamma_e = -0.65
gamma_z = 3.47
z0 = 0.6


__all__ = ("monte_carlo_bh_acc_rate",)


def eddington_ratio_distribution(redshift, npts=1000):
    """Power law model for the PDF of the Eddington ratio.

    Model is based on Aird, Coil, et al. (2011), arXiv:1107.4368.

    Parameters
    ----------
    redshift : float

    npts : int, optional
        Length of the lookup table used to define the Eddington ratio PDF

    Returns
    -------
    rate_table : ndarray
        Numpy array of shape (npts, )

    pdf_table : ndarray
        Numpy array of shape (npts, )

    Notes
    -----
    The returned ``rate_table`` and ``pdf_table`` will be used as inputs to the
    monte_carlo_from_cdf_lookup function to generate populations.

    Examples
    --------
    >>> redshift = 0.05
    >>> rate_table, pdf_table = eddington_ratio_distribution(redshift)
    """
    z = np.atleast_1d(redshift)
    msg = "Input ``redshift`` argument to eddington_ratio_distribution function " "must be a single float"
    assert len(z) == 1, msg
    redshift = float(max(z[0], 0.0))
    redshift = min(redshift, 5.0)  # No redshift evolution beyond z>5
    rate_table = np.logspace(-4, 0, npts)
    return (
        rate_table,
        A * (((1.0 + redshift) / (1.0 + z0)) ** gamma_z) * rate_table**gamma_e,
    )


def monte_carlo_eddington_ratio(redshift, sfr_percentile):
    """
    Monte Carlo realization of the Eddington ratio based on the power law model
    from Aird, Coil & Georgakakis (2017), arXiv:1705.01132.

    Parameters
    ----------
    redshift : float
        Median redshift of the sample

    sfr_percentile : ndarray
        Numpy array of shape (ngals, ) storing the SFR rank-order percentile
        at fixed stellar mass.

    Returns
    -------
    eddington_ratio : ndarray
        Numpy array of shape (ngals, ) storing the black hole accretion rate
        divided by the Eddington limit.

    Examples
    --------
    >>> redshift = 0.4
    >>> ngals = int(1e4)
    >>> sfr_percentile = np.random.uniform(0, 1, ngals)
    >>> eddington_ratio = monte_carlo_eddington_ratio(redshift, sfr_percentile)

    """
    redshift = np.atleast_1d(redshift)
    msg = "monte_carlo_eddington_ratio only accepts " "a single float for ``redshift`` argument"
    assert len(redshift) == 1, msg

    rate_table, prob_table = eddington_ratio_distribution(redshift[0])
    mc_input = 1 - sfr_percentile
    return monte_carlo_from_cdf_lookup(rate_table, prob_table, mc_input)


def monte_carlo_from_cdf_lookup(x_table, y_table, mc_input):
    return np.interp(np.atleast_1d(mc_input), x_table, y_table)


def monte_carlo_bh_acc_rate(redshift, black_hole_mass, sfr_percentile):
    """
    Monte Carlo realization of black hole accretion rate
    based on the power law model from Aird, Coil & Georgakakis (2017),
    arXiv:1705.01132.

    Parameters
    ----------
    redshift : float
        Median redshift of the sample

    black_hole_mass : ndarray
        Numpy array of shape (ngals, ) storing the black hole mass
        in units of Msun assuming h=0.7

    sfr_percentile : ndarray
        Numpy array of shape (ngals, ) storing the SFR rank-order percentile
        at fixed stellar mass.

    Returns
    -------
    eddington_ratio : ndarray
        Numpy array of shape (ngals, ) storing dimensionless Eddington ratio

    accretion_rate : ndarray
        Numpy array of shape (ngals, ) storing dM_bh/dt
        in units of Msun/yr assuming h=0.7

    Examples
    --------
    >>> redshift = 0.4
    >>> ngals = int(1e4)
    >>> sfr_percentile = np.random.uniform(0, 1, ngals)
    >>> black_hole_mass = 10**np.random.uniform(6, 11, ngals)
    >>> args = redshift, black_hole_mass, sfr_percentile
    >>> edd_ratio, acc_rate = monte_carlo_bh_acc_rate(*args)

    """
    redshift = np.atleast_1d(redshift)
    msg = "monte_carlo_bh_acc_rate only accepts " "a single float for ``redshift`` argument"
    assert len(redshift) == 1, msg

    eddington_ratio = monte_carlo_eddington_ratio(redshift, sfr_percentile)
    eddington_rate = black_hole_mass * 2.2e-8
    accretion_rate = eddington_ratio * eddington_rate
    return eddington_ratio, accretion_rate
