""" """

import numpy as np
from dsps.data_loaders.load_ssp_data import load_ssp_templates

from .. import precompute_ssp_phot as psp
from ..phot_utils import load_dsps_lsst_tcurves


def test_get_interpolated_tcurves():
    tcurves = load_dsps_lsst_tcurves()
    ssp_data = load_ssp_templates()

    z_obs = 1.0
    new_tcurves = psp.get_redshifted_and_interpolated_tcurves(
        tcurves, ssp_data.ssp_wave, z_obs
    )
    X, Y = psp.get_tcurve_matrix_from_tcurves(new_tcurves)


def test_get_ssp_restflux_table():
    tcurves = load_dsps_lsst_tcurves()
    n_filters = len(tcurves)
    ssp_data = load_ssp_templates()
    n_met, n_age = ssp_data.ssp_flux.shape[0:2]
    z_kcorrect = 0.1
    ssp_restflux_table = psp.get_ssp_restflux_table(ssp_data, tcurves, z_kcorrect)
    assert np.all(np.isfinite(ssp_restflux_table))
    assert ssp_restflux_table.shape == (n_filters, n_met, n_age)
