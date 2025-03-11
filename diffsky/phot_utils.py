"""
"""

import os

import numpy as np
from dsps.data_loaders import load_filter_data
from dsps.data_loaders.retrieve_fake_fsps_data import (
    load_fake_filter_transmission_curves,
)
from dsps.dust.utils import get_filter_effective_wavelength

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
    tcurve_u, tcurve_g, tcurve_r, tcurve_i, tcurve_z, tcurve_y = lsst_tcurves

    tcurve_u = tcurve_u._replace(
        transmission=np.interp(ssp_wave, tcurve_u.wave, tcurve_u.transmission)
    )
    tcurve_g = tcurve_g._replace(
        transmission=np.interp(ssp_wave, tcurve_g.wave, tcurve_g.transmission)
    )
    tcurve_r = tcurve_r._replace(
        transmission=np.interp(ssp_wave, tcurve_r.wave, tcurve_r.transmission)
    )
    tcurve_i = tcurve_i._replace(
        transmission=np.interp(ssp_wave, tcurve_i.wave, tcurve_i.transmission)
    )
    tcurve_z = tcurve_z._replace(
        transmission=np.interp(ssp_wave, tcurve_z.wave, tcurve_z.transmission)
    )
    tcurve_y = tcurve_y._replace(
        transmission=np.interp(ssp_wave, tcurve_y.wave, tcurve_y.transmission)
    )

    tcurve_u = tcurve_u._replace(wave=ssp_wave)
    tcurve_g = tcurve_g._replace(wave=ssp_wave)
    tcurve_r = tcurve_r._replace(wave=ssp_wave)
    tcurve_i = tcurve_i._replace(wave=ssp_wave)
    tcurve_z = tcurve_z._replace(wave=ssp_wave)
    tcurve_y = tcurve_y._replace(wave=ssp_wave)

    tcurves = list((tcurve_u, tcurve_g, tcurve_r, tcurve_i, tcurve_z, tcurve_y))
    return tcurves


def interp_tcurve(tcurve_orig, ssp_data):
    """"""
    indx_insert = np.searchsorted(ssp_data.ssp_wave, tcurve_orig.wave)
    xarr = np.insert(ssp_data.ssp_wave, indx_insert, tcurve_orig.wave)
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
