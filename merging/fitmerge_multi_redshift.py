# flake8: noqa: E402
"""Functions related to calibrating the merging model."""

from jax import config

config.update("jax_enable_x64", True)

from dsps.utils import _tw_cuml_kern
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from .merging_model import merge_model_with_preprocessing


@jjit
def triweighted_histogram(X, sigma, lobins, bin_width):
    """Triweighted histogram

    Parameters
    ----------
    X : float
        Value to evalute

    sigma : float
        Width of triweight kernel

    lobins : float
        Lower bin edges

    bin_width : float
        Bin width

    Returns
    -------
    hist : float
        Triweight histogram

    """
    last_cdf = _tw_cuml_kern(lobins, X, sigma)
    new_cdf = _tw_cuml_kern(lobins + bin_width, X, sigma)
    weight = new_cdf - last_cdf
    hist = jnp.sum(weight)
    return hist


@jjit
def weighted_tw_histogram(X, sigma, y, lobins, bin_width):
    """Triweighted histogram

    Parameters
    ----------
    X : float
        Value to evalute

    sigma : float
        Width of triweight kernel

    y : float
        weights to sum

    lobins : float
        Lower bin edges

    bin_width : float
        Bin width

    Returns
    -------
    hist : float
        Triweight histogram

    """
    last_cdf = _tw_cuml_kern(lobins, X, sigma)
    new_cdf = _tw_cuml_kern(lobins + bin_width, X, sigma)
    weight = new_cdf - last_cdf
    hist = jnp.sum(weight * y)
    return hist


mapped_weighted_tw_histogram = jjit(
    vmap(weighted_tw_histogram, in_axes=(None, None, None, 0, None))
)


@jjit
def scalar_smf(logM, lobins, a_interest, sigma, bin_width, comoving_volume):
    """Calculate a single bin of the differentiable (triweighted) SMF

    Parameters
    ----------
    logM : float
        log stellar mass

    lobins: float
        lower bin edge

    a_interest : float
        scale factor of interest

    sigma : float
        width of triweight

    bin_width : float
        width of stellar mass bin

    comoving_volume : float
        comoving volume of box

    Returns
    -------

    hist : float
        a single bin of the stellar mass function

    """
    physical_volume = comoving_volume * (a_interest**3)
    hist = (
        triweighted_histogram(logM, sigma, lobins, bin_width)
        / bin_width
        / physical_volume
    )
    return hist


mapped_histogram = vmap(scalar_smf, in_axes=(None, 0, None, None, None, None))


# Conditional stellar mass function
def csmf_sats_cens_all(
    all_sm_no_zero, host_lgmp, upid, a, lobins, sigma, bin_width, comoving_volume
):
    """Conditional stellar mass function

    Parameters
    ----------
    all_sm_no_zero : float
        stellar mass

    host_lgmp: float
        log peak mass of host

    mock : Table
        Table of galaxy info

    a : float
        scale factor

    lobins : float
        lower bin edge

    sigma : float
        width of triweight

    bin_width : float
        width of stellar mass bin

    comoving_volume : float
        comoving volume of box

    Returns
    -------

    smf : list of shape (9,)
        satellite, central, and total stellar mass function in 3 halo mass bins

    """
    low = [12, 13, 14]
    high = [13, 14, 18]
    smf = []

    for i in range(3):
        smf_sats = mapped_histogram(
            jnp.log10(
                all_sm_no_zero[
                    (host_lgmp > low[i]) & (host_lgmp < high[i]) & (upid != -1)
                ]
            ),
            lobins,
            a,
            sigma,
            bin_width,
            comoving_volume,
        )
        smf_cens = mapped_histogram(
            jnp.log10(
                all_sm_no_zero[
                    (host_lgmp > low[i]) & (host_lgmp < high[i]) & (upid == -1)
                ]
            ),
            lobins,
            a,
            sigma,
            bin_width,
            comoving_volume,
        )
        smf_all = mapped_histogram(
            jnp.log10(all_sm_no_zero[(host_lgmp > low[i]) & (host_lgmp < high[i])]),
            lobins,
            a,
            sigma,
            bin_width,
            comoving_volume,
        )
        smf.append(smf_sats)
        smf.append(smf_cens)
        smf.append(smf_all)

    return smf


def condition(host_lgmp):
    """Condition for CSMF

    Parameters
    ----------
    host_lgmp : array of shape (n_gals,)
        log peak mass of host

    Returns
    -------

    condition12 : array of shape (n_gals,)
        indices of galaxies that meet this host mass condition

    condition13 : array of shape (n_gals,)
        indices of galaxies that meet this host mass condition

    condition14 : array of shape (n_gals,)
        indices of galaxies that meet this host mass condition

    """
    condition12 = jnp.argwhere((host_lgmp > 12) & (host_lgmp < 13))
    condition13 = jnp.argwhere((host_lgmp > 13) & (host_lgmp < 14))
    condition14 = jnp.argwhere((host_lgmp > 14))
    return condition12, condition13, condition14


# Conditional stellar mass function
def csmf(all_sm_no_zero, host_lgmp, a, lobins, sigma, bin_width, comoving_volume):

    low = [12, 13, 14]
    high = [13, 14, 18]
    smf = []

    for i in range(3):
        smf_all = mapped_histogram(
            jnp.log10(all_sm_no_zero[(host_lgmp > low[i]) & (host_lgmp < high[i])]),
            lobins,
            a,
            sigma,
            bin_width,
            comoving_volume,
        )
        smf.append(smf_all)

    return smf


@jjit
def model_histogram(
    merging_u_params,
    logMpeak_penultimate_infall,
    logMpeak_ultimate_infall,
    logMhost_penultimate_infall,
    logMhost_ultimate_infall,
    tinterest,
    t_penultimate_infall,
    t_ultimate_infall,
    upids,
    sfr,
    penultimate_dump,
    ultimate_dump,
    MC,
    dT,
    i_interest,
    a_interest,
    sigma,
    bin_width,
    comoving_volume,
    lobins,
):

    SFR = merge_model_with_preprocessing(
        merging_u_params,
        logMpeak_penultimate_infall,
        logMpeak_ultimate_infall,
        logMhost_penultimate_infall,
        logMhost_ultimate_infall,
        tinterest,
        t_penultimate_infall,
        t_ultimate_infall,
        upids,
        sfr,
        penultimate_dump,
        ultimate_dump,
        MC,
    )

    M = jnp.cumsum(SFR * dT, axis=1)

    M_interest = M[:, i_interest]

    hist = (
        scalar_smf(
            jnp.log10(M_interest), lobins, a_interest, sigma, bin_width, comoving_volume
        )
        + 1e-10
    )

    return hist


@jjit
def conditional_model_histogram(
    merging_u_params,
    logMpeak_penultimate_infall,
    logMpeak_ultimate_infall,
    logMhost_penultimate_infall,
    logMhost_ultimate_infall,
    tinterest,
    t_penultimate_infall,
    t_ultimate_infall,
    upids,
    sfr,
    penultimate_dump,
    ultimate_dump,
    MC,
    dT,
    i_interest,
    a_interest,
    sigma,
    bin_width,
    comoving_volume,
    lobins,
    keep,
):

    SFR = merge_model_with_preprocessing(
        merging_u_params,
        logMpeak_penultimate_infall,
        logMpeak_ultimate_infall,
        logMhost_penultimate_infall,
        logMhost_ultimate_infall,
        tinterest,
        t_penultimate_infall,
        t_ultimate_infall,
        upids,
        sfr,
        penultimate_dump,
        ultimate_dump,
        MC,
    )

    M = jnp.cumsum(SFR * dT, axis=1)

    M_interest = M[:, i_interest]

    M_keep = M_interest[keep] + 1e-10

    hist = (
        scalar_smf(
            jnp.log10(M_keep), lobins, a_interest, sigma, bin_width, comoving_volume
        )
        + 1e-10
    )

    return hist


mapped_model_csmf = vmap(
    conditional_model_histogram,
    in_axes=(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        0,
        None,
    ),
)


@jjit
def model_mass(
    merging_u_params,
    logMpeak_penultimate_infall,
    logMpeak_ultimate_infall,
    logMhost_penultimate_infall,
    logMhost_ultimate_infall,
    tinterest,
    t_penultimate_infall,
    t_ultimate_infall,
    upids,
    sfr,
    penultimate_dump,
    ultimate_dump,
    MC,
    dT,
    i_interest,
):

    SFR = merge_model_with_preprocessing(
        merging_u_params,
        logMpeak_penultimate_infall,
        logMpeak_ultimate_infall,
        logMhost_penultimate_infall,
        logMhost_ultimate_infall,
        tinterest,
        t_penultimate_infall,
        t_ultimate_infall,
        upids,
        sfr,
        penultimate_dump,
        ultimate_dump,
        MC,
    )

    M = jnp.cumsum(SFR * dT, axis=1)

    M_interest = M[:, i_interest]

    return M_interest


@jjit
def sat_frac(
    merging_u_params,
    logMpeak_penultimate_infall,
    logMpeak_ultimate_infall,
    logMhost_penultimate_infall,
    logMhost_ultimate_infall,
    tinterest,
    t_penultimate_infall,
    t_ultimate_infall,
    upids,
    sfr,
    penultimate_dump,
    ultimate_dump,
    MC,
    dT,
    i_interest,
    bins,
    bin_width,
):

    stellar_mass = model_mass(
        merging_u_params,
        logMpeak_penultimate_infall,
        logMpeak_ultimate_infall,
        logMhost_penultimate_infall,
        logMhost_ultimate_infall,
        tinterest,
        t_penultimate_infall,
        t_ultimate_infall,
        upids,
        sfr,
        penultimate_dump,
        ultimate_dump,
        MC,
        dT,
        i_interest,
    )

    logsm = jnp.log10(stellar_mass)
    sats_logsm = jnp.where(upids != -1, logsm, 0)
    sm = mapped_weighted_tw_histogram(logsm, 0.05, 1, bins, bin_width)
    sat_sm = mapped_weighted_tw_histogram(sats_logsm, 0.05, 1, bins, bin_width)
    sat_frac = sat_sm / sm

    return sat_frac
