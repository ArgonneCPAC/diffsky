""" """

import os

from dsps.data_loaders import load_filter_data
from dsps.data_loaders.retrieve_fake_fsps_data import (
    load_fake_filter_transmission_curves,
)

try:
    DEFAULT_DSPS_DRN = os.environ["DSPS_DRN"]
except KeyError:
    DEFAULT_DSPS_DRN = ""


def load_lsst_curves(drn_ssp_data=None):
    if drn_ssp_data is None:
        lsst_tcurves = load_fake_lsst_tcurves()
        print("Using fake LSST transmission curves since input drn_ssp_data=None")
    else:

        try:
            lsst_tcurves = load_dsps_lsst_tcurves(drn_ssp_data=drn_ssp_data)
        except (AssertionError, ImportError, OSError, ValueError):
            msg = f"Input drn does not contain DSPS data: drn_ssp_data=`{drn_ssp_data}`"
            raise ValueError(msg)

    return lsst_tcurves


def load_dsps_lsst_tcurves(drn_ssp_data=DEFAULT_DSPS_DRN):
    """"""
    drn_filters = os.path.join(drn_ssp_data, "filters")
    tcurve_u = load_filter_data.load_transmission_curve(
        bn_pat="lsst_u*", drn=drn_filters
    )
    tcurve_g = load_filter_data.load_transmission_curve(
        bn_pat="lsst_g*", drn=drn_filters
    )
    tcurve_r = load_filter_data.load_transmission_curve(
        bn_pat="lsst_r*", drn=drn_filters
    )
    tcurve_i = load_filter_data.load_transmission_curve(
        bn_pat="lsst_i*", drn=drn_filters
    )
    tcurve_z = load_filter_data.load_transmission_curve(
        bn_pat="lsst_z*", drn=drn_filters
    )
    tcurve_y = load_filter_data.load_transmission_curve(
        bn_pat="lsst_y*", drn=drn_filters
    )
    tcurves = list((tcurve_u, tcurve_g, tcurve_r, tcurve_i, tcurve_z, tcurve_y))
    return tcurves


def load_fake_lsst_tcurves():
    """"""
    _res = load_fake_filter_transmission_curves()
    wave, u, g, r, i, z, y = _res
    tcurve_u = load_filter_data.TransmissionCurve(wave, u)
    tcurve_g = load_filter_data.TransmissionCurve(wave, g)
    tcurve_r = load_filter_data.TransmissionCurve(wave, r)
    tcurve_i = load_filter_data.TransmissionCurve(wave, i)
    tcurve_z = load_filter_data.TransmissionCurve(wave, z)
    tcurve_y = load_filter_data.TransmissionCurve(wave, y)
    tcurves = list((tcurve_u, tcurve_g, tcurve_r, tcurve_i, tcurve_z, tcurve_y))
    return tcurves
