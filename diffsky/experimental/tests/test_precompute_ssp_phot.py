""" """

import numpy as np
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders.retrieve_fake_fsps_data import load_fake_ssp_data

from .. import precompute_ssp_phot as psp
from ..phot_utils import load_fake_lsst_tcurves


def test_get_interpolated_tcurves():
    tcurves = load_fake_lsst_tcurves()
    ssp_data = load_fake_ssp_data()

    z_obs = 1.0
    new_tcurves = psp.get_redshifted_and_interpolated_tcurves(
        tcurves, ssp_data.ssp_wave, z_obs
    )
    X, Y = psp.get_tcurve_matrix_from_tcurves(new_tcurves)


def test_get_ssp_restflux_table():
    tcurves = load_fake_lsst_tcurves()
    n_filters = len(tcurves)
    ssp_data = load_fake_ssp_data()
    n_met, n_age = ssp_data.ssp_flux.shape[0:2]
    z_kcorrect = 0.1
    ssp_restflux_table = psp.get_ssp_restflux_table(ssp_data, tcurves, z_kcorrect)
    assert np.all(np.isfinite(ssp_restflux_table))
    assert ssp_restflux_table.shape == (n_filters, n_met, n_age)


def test_get_ssp_obsflux_table():
    tcurves = load_fake_lsst_tcurves()
    n_filters = len(tcurves)
    ssp_data = load_fake_ssp_data()
    n_met, n_age = ssp_data.ssp_flux.shape[0:2]
    z_obs = 1.0
    ssp_obsflux_table = psp.get_ssp_obsflux_table(
        ssp_data, tcurves, z_obs, DEFAULT_COSMOLOGY
    )
    assert np.all(np.isfinite(ssp_obsflux_table))
    assert ssp_obsflux_table.shape == (n_filters, n_met, n_age)
