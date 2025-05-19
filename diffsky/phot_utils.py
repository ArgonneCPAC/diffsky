"""
"""

import os

import numpy as np
from dsps.data_loaders import load_filter_data
from dsps.data_loaders.retrieve_fake_fsps_data import (
    load_fake_filter_transmission_curves,
)
from dsps.dust.utils import get_filter_effective_wavelength
from dsps.photometry import utils as phu

try:
    DEFAULT_DSPS_DRN = os.environ["DSPS_DRN"]
except KeyError:
    DEFAULT_DSPS_DRN = ""


def load_interpolated_lsst_curves(ssp_wave, drn_ssp_data=None):
    lsst_tcurves_nointerp = load_lsst_curves(drn_ssp_data=drn_ssp_data)
    lsst_tcurves_interp = interpolate_lsst_tcurves(lsst_tcurves_nointerp, ssp_wave)
    return lsst_tcurves_interp, lsst_tcurves_nointerp


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


def interpolate_lsst_tcurves(lsst_tcurves, ssp_wave):
    wave_filters = [x.wave for x in lsst_tcurves]
    trans_filters = [x.transmission for x in lsst_tcurves]
    wave_matrix, trans_matrix = phu.interpolate_filter_trans_curves(
        wave_filters, trans_filters, len(ssp_wave)
    )
    collector = []
    n_filters = wave_matrix.shape[0]
    for i in range(n_filters):
        wave, trans = wave_matrix[i, :], trans_matrix[i, :]
        tcurve = load_filter_data.TransmissionCurve(wave, trans)
        collector.append(tcurve)
    return collector


def interp_tcurve(tcurve_orig, ssp_wave):
    """"""
    indx_insert = np.searchsorted(ssp_wave, tcurve_orig.wave)
    xarr = np.insert(ssp_wave, indx_insert, tcurve_orig.wave)
    tcurve_new_trans = np.interp(xarr, tcurve_orig.wave, tcurve_orig.transmission)
    msk = (xarr >= tcurve_orig.wave[0]) & (xarr <= tcurve_orig.wave[-1])
    tcurve_new_wave = xarr[msk]
    tcurve_new_trans = tcurve_new_trans[msk]
    tcurve_new = load_filter_data.TransmissionCurve(tcurve_new_wave, tcurve_new_trans)
    return tcurve_new


def get_wave_eff_from_tcurves(tcurves, z_obs):
    wave_eff_arr = []
    for tcurve in tcurves:
        waveff = get_filter_effective_wavelength(
            tcurve.wave, tcurve.transmission, z_obs
        )
        wave_eff_arr.append(waveff)
    wave_eff_arr = np.array(wave_eff_arr)
    return wave_eff_arr
