"""Module implements the get_ssp_restflux_table and get_ssp_obsflux_table
functions for precomputing SSP magnitude tables"""

import numpy as np
from dsps.photometry import photometry_kernels as pk
from jax import jit as jjit
from jax import vmap

_SSP = (None, 0, None, None)
_calc_rest_flux_ssp = jjit(vmap(vmap(pk.calc_rest_flux, in_axes=_SSP), in_axes=_SSP))
_B = (None, None, 0, 0)
_calc_rest_flux_ssp_bands = jjit(vmap(_calc_rest_flux_ssp, in_axes=_B))
EPS = 1e-5


def get_ssp_restflux_table(ssp_data, tcurves, z_kcorrect):
    """Compute restframe broadband flux of a collection of SSP SEDs.

    This function is typically used to precompute absolute magnitudes of SSPs
    in population-level forward modeling applications, running large batches of MCMCs
    of individual galaxy photometry, and other performance-critical applications.

    Parameters
    ----------
    ssp_data : namedtuple
        Object returned by dsps.data_loaders.load_ssp_templates
        Tuple entries: ('ssp_lgmet', 'ssp_lg_age_gyr', 'ssp_wave', 'ssp_flux')

    tcurves : list of namedtuples
        Each element is an instance of dsps.data_loaders.load_transmission_curve
        Tuple entries: ('wave', 'transmission')

    z_kcorrect : float
        Redshift to which the SSP SEDs are k-corrected

    Returns
    -------
    ssp_restflux_table : ndarray, shape (n_filters, n_met, n_age)
        ssp_restmag_table = -2.5log10(ssp_restflux_table)

    """
    tcurves = get_redshifted_and_interpolated_tcurves(
        tcurves, ssp_data.ssp_wave, z_kcorrect
    )
    wave_filters, trans_filters = get_tcurve_matrix_from_tcurves(tcurves)
    ssp_restflux_table = _calc_rest_flux_ssp_bands(
        ssp_data.ssp_wave, ssp_data.ssp_flux, wave_filters, trans_filters
    )
    return ssp_restflux_table


def get_ssp_obsflux_table(ssp_data, tcurves, z_obs, cosmo_params):
    """Compute observer-frame broadband flux of a collection of SSP SEDs.

    This function is typically used to precompute apparent magnitudes of SSPs
    in population-level forward modeling applications, running large batches of MCMCs
    of individual galaxy photometry, and other performance-critical applications.

    Parameters
    ----------
    ssp_data : namedtuple
        Object returned by dsps.data_loaders.load_ssp_templates
        Tuple entries: ('ssp_lgmet', 'ssp_lg_age_gyr', 'ssp_wave', 'ssp_flux')

    tcurves : list of namedtuples
        Each element is an instance of dsps.data_loaders.load_transmission_curve
        Tuple entries: ('wave', 'transmission')

    z_obs : float
        Observed redshift of the galaxy

    cosmo_params : namedtuple
        Instance of dsps.cosmology.CosmoParams
        Tuple entries: ('Om0', 'w0', 'wa', 'h')

    Returns
    -------
    ssp_obsflux_table : ndarray, shape (n_filters, n_met, n_age)
        ssp_obsmag_table = -2.5log10(ssp_obsflux_table)

    """
    ssp_flux_table = get_ssp_restflux_table(ssp_data, tcurves, z_obs)
    dimming = pk._cosmological_dimming(
        z_obs, cosmo_params.Om0, cosmo_params.w0, cosmo_params.wa, cosmo_params.h
    )
    ssp_mag_table = -2.5 * np.log10(ssp_flux_table)
    ssp_obsmag_table = ssp_mag_table + dimming
    ssp_obsflux_table = 10 ** (-0.4 * ssp_obsmag_table)
    return ssp_obsflux_table


def interp_tcurve_to_ssp(tcurve_wave, tcurve_trans, ssp_wave):
    """"""
    cuml_trans = np.cumsum(tcurve_trans)
    cuml_trans = cuml_trans / cuml_trans[-1]
    indx_x_min = np.searchsorted(cuml_trans, EPS)
    indx_x_max = np.searchsorted(cuml_trans, 1.0 - EPS)
    x_min = tcurve_wave[indx_x_min]
    x_max = tcurve_wave[indx_x_max]

    indx_insert = np.searchsorted(ssp_wave, tcurve_wave)
    x0 = np.insert(ssp_wave, indx_insert, tcurve_wave)
    x1 = np.unique(x0)
    y1 = np.interp(x1, tcurve_wave, tcurve_trans)

    msk = (x1 < x_min) | (x1 > x_max)
    y1 = np.where(msk, 0.0, y1)

    x_out = x1[~msk]
    y_out = y1[~msk]

    return x_out, y_out


def _pad_tcurve(tcurve_wave, tcurve_trans, n):
    n_pad = n - tcurve_wave.size
    dx = np.diff(tcurve_wave).min() / 100.0
    x_pad = tcurve_wave[0] - np.arange(1, n_pad + 1, 1) * dx
    tcurve_wave_out = np.concatenate((x_pad, tcurve_wave))
    tcurve_trans_out = np.concatenate((np.zeros(n_pad), tcurve_trans))
    return tcurve_wave_out, tcurve_trans_out


def redshift_tcurve(tcurve, redshift):
    x = tcurve.wave / (1.0 + redshift)
    tcurve = tcurve._make((x, tcurve.transmission))
    return tcurve


def get_redshifted_and_interpolated_tcurves(tcurves, ssp_wave, redshift):
    """"""
    tcurves = [redshift_tcurve(tcurve, redshift) for tcurve in tcurves]

    tcurves_refined = []
    for tcurve in tcurves:
        args = tcurve.wave, tcurve.transmission, ssp_wave
        x, y = interp_tcurve_to_ssp(*args)
        new_tcurve = tcurves[0]._make((x, y))
        tcurves_refined.append(new_tcurve)

    n = np.max([x.wave.size for x in tcurves_refined])
    interpolated_tcurves = []
    for tcurve in tcurves_refined:
        args = tcurve.wave, tcurve.transmission, n
        x, y = _pad_tcurve(*args)
        new_tcurve = tcurves[0]._make((x, y))
        interpolated_tcurves.append(new_tcurve)

    return interpolated_tcurves


def get_tcurve_matrix_from_tcurves(tcurves):
    X = np.array([x.wave for x in tcurves])
    Y = np.array([x.transmission for x in tcurves])
    return X, Y
