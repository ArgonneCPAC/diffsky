""" """

import numpy as np
from jax import jit as jjit

EPS = 1e-5


def interp_tcurve_to_ssp(tcurve_wave, tcurve_trans, ssp_wave, redshift):
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


def get_interpolated_tcurves(tcurves, ssp_wave, redshift):
    """"""
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


@jjit
def get_ssp_restflux_table(
    ssp_wave, ssp_fluxes, wave_matrix, transmission_matrix, z_kcorrect
):
    pass


@jjit
def get_ssp_obsflux_table(
    ssp_wave, ssp_fluxes, wave_matrix, transmission_matrix, z_obs, cosmo_params
):
    pass
