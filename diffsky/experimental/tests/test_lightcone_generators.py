""""""

from collections import namedtuple

import numpy as np
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.data_loaders.defaults import TransmissionCurve
from jax import random as jran

from ...data_loaders import load_ssp_data
from .. import lightcone_generators as lcg
from .. import mc_phot
from . import test_mc_phot as tmcp


def _get_weighted_lc_halos_photdata_for_unit_testing(num_halos=75):
    ran_key = jran.key(0)

    lgmp_min, lgmp_max = 10.0, 15.0
    z_min, z_max = 0.1, 3.0
    sky_area_degsq = 100.0

    EMLINE_NAMES = ("Halpha", "OII", "OIII", "NII")
    ssp_data = load_ssp_data.load_fake_ssp_data(emline_names=EMLINE_NAMES)

    _res = retrieve_fake_fsps_data.load_fake_filter_transmission_curves()
    wave, u, g, r, i, z, y = _res

    tcurve_list = [TransmissionCurve(wave, x) for x in (u, g, r, i, z, y)]
    names = [f"lsst_{x}" for x in ("u", "g", "r", "i", "z", "y")]
    TransmissionCurves = namedtuple("TransmissionCurves", names)
    tcurves = TransmissionCurves(*tcurve_list)

    n_z_phot_table = 15
    z_phot_table = 10 ** np.linspace(np.log10(z_min), np.log10(z_max), n_z_phot_table)

    args = (
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    lc_data = lcg.weighted_lc_halos_photdata(*args)

    return lc_data, tcurves


def _get_weighted_lc_photdata_for_unit_testing(num_halos=75):
    ran_key = jran.key(0)

    lgmp_min, lgmp_max = 10.0, 15.0
    z_min, z_max = 0.1, 3.0
    sky_area_degsq = 100.0

    EMLINE_NAMES = ("Halpha", "OII", "OIII", "NII")
    ssp_data = load_ssp_data.load_fake_ssp_data(emline_names=EMLINE_NAMES)

    _res = retrieve_fake_fsps_data.load_fake_filter_transmission_curves()
    wave, u, g, r, i, z, y = _res

    tcurve_list = [TransmissionCurve(wave, x) for x in (u, g, r, i, z, y)]
    names = [f"lsst_{x}" for x in ("u", "g", "r", "i", "z", "y")]
    TransmissionCurves = namedtuple("TransmissionCurves", names)
    tcurves = TransmissionCurves(*tcurve_list)

    n_z_phot_table = 15
    z_phot_table = 10 ** np.linspace(np.log10(z_min), np.log10(z_max), n_z_phot_table)

    args = (
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    lc_data = lcg.weighted_lc_photdata(*args)

    return lc_data, tcurves


def test_weighted_lc_halos_photdata():
    lc_data, tcurves = _get_weighted_lc_halos_photdata_for_unit_testing()

    for field in lcg.LCHalosData._fields:
        assert hasattr(lc_data, field)

    for arr in lc_data.mah_params:
        assert np.all(np.isfinite(arr))

    assert np.all(lc_data.logmp0 >= lc_data.logmp_obs)

    n_bands = len(tcurves)
    n_met, n_age = lc_data.ssp_data.ssp_flux.shape[:-1]
    n_z_phot_table = lc_data.z_phot_table.size

    correct_shape = (n_z_phot_table, n_bands, n_met, n_age)
    assert lc_data.precomputed_ssp_mag_table.shape == correct_shape
    assert np.all(lc_data.precomputed_ssp_mag_table > 0)

    assert lc_data.wave_eff_table.shape == (n_z_phot_table, n_bands)
    assert np.all(lc_data.wave_eff_table > 100)
    assert np.all(lc_data.wave_eff_table < 40_000)

    assert hasattr(lc_data, "precomputed_ssp_lineflux_cgs_table")
    assert hasattr(lc_data, "line_wave_table")

    EMLINE_NAMES = lc_data.ssp_data.emlines._fields
    n_lines = len(EMLINE_NAMES)
    assert lc_data.precomputed_ssp_lineflux_cgs_table.shape == (n_lines, n_met, n_age)
    assert lc_data.line_wave_table.shape == (n_lines,)

    ran_key = jran.key(1)
    phot_kern_results = mc_phot.mc_lc_phot(ran_key, lc_data)
    keys = list(phot_kern_results.keys())
    phot_kern_results = namedtuple("Results", keys)(**phot_kern_results)

    tmcp.check_phot_kern_results(phot_kern_results)


def test_weighted_lc_photdata():
    num_halos = 75
    lc_data, tcurves = _get_weighted_lc_photdata_for_unit_testing(num_halos=num_halos)
    n_tot = lc_data.z_obs.size
    shape_ntot_keys = (
        "z_obs",
        "t_obs",
        "logmp_obs",
        "logmp0",
        "t_infall",
        "logmp_infall",
        "logmhost_infall",
        "is_central",
        "halo_indx",
    )
    for lc_key in shape_ntot_keys:
        arr = getattr(lc_data, lc_key)
        assert arr.shape == (n_tot,), f"lc_data.{lc_key} has the wrong shape"
        assert np.all(np.isfinite(arr)), f"lc_data.{lc_key} has NaNs"

    for field in lcg.LCData._fields:
        assert hasattr(lc_data, field)

    for arr in lc_data.mah_params:
        assert np.all(np.isfinite(arr))

    ran_key = jran.key(1)
    phot_kern_results = mc_phot.mc_lc_phot(ran_key, lc_data)
    keys = list(phot_kern_results.keys())
    phot_kern_results = namedtuple("Results", keys)(**phot_kern_results)

    tmcp.check_phot_kern_results(phot_kern_results)
