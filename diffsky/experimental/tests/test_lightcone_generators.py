""""""

import numpy as np
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.data_loaders.defaults import TransmissionCurve
from jax import random as jran

from ...data_loaders import load_ssp_data
from .. import lightcone_generators as lcg


def test_mc_weighted_lightcone_data():
    ran_key = jran.key(0)

    num_halos = 75
    lgmp_min, lgmp_max = 10.0, 15.0
    z_min, z_max = 0.1, 3.0
    sky_area_degsq = 100.0

    ssp_data = load_ssp_data.load_fake_ssp_data()

    _res = retrieve_fake_fsps_data.load_fake_filter_transmission_curves()
    wave, u, g, r, i, z, y = _res

    tcurves = [TransmissionCurve(wave, x) for x in (u, g, r, i, z, y)]

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
    lc_data = lcg.mc_weighted_lightcone_data(*args)

    for field in lcg.LCData._fields:
        assert hasattr(lc_data, field)

    for arr in lc_data.mah_params:
        assert np.all(np.isfinite(arr))

    assert np.all(lc_data.logmp0 >= lc_data.logmp_obs)

    n_bands = len(tcurves)
    n_met, n_age = ssp_data.ssp_flux.shape[:-1]
    correct_shape = (n_z_phot_table, n_bands, n_met, n_age)
    assert lc_data.precomputed_ssp_mag_table.shape == correct_shape
    assert np.all(lc_data.precomputed_ssp_mag_table > 0)

    assert lc_data.wave_eff_table.shape == (n_z_phot_table, n_bands)
    assert np.all(lc_data.wave_eff_table > 100)
    assert np.all(lc_data.wave_eff_table < 40_000)
