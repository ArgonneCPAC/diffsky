""" """

from dsps.data_loaders.load_ssp_data import load_ssp_templates

from .. import precompute_ssp_phot as psp
from ..phot_utils import load_dsps_lsst_tcurves


def test_get_interpolated_tcurves():
    lsst_tcurves = load_dsps_lsst_tcurves()
    ssp_data = load_ssp_templates()

    z_obs = 1.0
    new_tcurves = psp.get_interpolated_tcurves(lsst_tcurves, ssp_data.ssp_wave, z_obs)
