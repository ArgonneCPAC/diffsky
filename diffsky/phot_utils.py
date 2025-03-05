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


def get_interpolated_lsst_tcurves(ssp_wave, drn_ssp_data=DEFAULT_DSPS_DRN):
    try:
        tcurve_u = load_filter_data.load_transmission_curve(
            bn_pat="lsst_u*", drn=drn_ssp_data
        )
        tcurve_g = load_filter_data.load_transmission_curve(
            bn_pat="lsst_g*", drn=drn_ssp_data
        )
        tcurve_r = load_filter_data.load_transmission_curve(
            bn_pat="lsst_r*", drn=drn_ssp_data
        )
        tcurve_i = load_filter_data.load_transmission_curve(
            bn_pat="lsst_i*", drn=drn_ssp_data
        )
        tcurve_z = load_filter_data.load_transmission_curve(
            bn_pat="lsst_z*", drn=drn_ssp_data
        )
        tcurve_y = load_filter_data.load_transmission_curve(
            bn_pat="lsst_y*", drn=drn_ssp_data
        )
    except (ImportError, OSError, ValueError, AssertionError):
        _res = load_fake_filter_transmission_curves()
        wave, u, g, r, i, z, y = _res
        tcurve_u = load_filter_data.TransmissionCurve(wave, u)
        tcurve_g = load_filter_data.TransmissionCurve(wave, g)
        tcurve_r = load_filter_data.TransmissionCurve(wave, r)
        tcurve_i = load_filter_data.TransmissionCurve(wave, i)
        tcurve_z = load_filter_data.TransmissionCurve(wave, z)
        tcurve_y = load_filter_data.TransmissionCurve(wave, y)

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


def get_wave_eff_from_tcurves(tcurves, z_obs):
    wave_eff_arr = []
    for tcurve in tcurves:
        waveff = get_filter_effective_wavelength(
            tcurve.wave, tcurve.transmission, z_obs
        )
        wave_eff_arr.append(waveff)
    wave_eff_arr = np.array(wave_eff_arr)
    return wave_eff_arr
