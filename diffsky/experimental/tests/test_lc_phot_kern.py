""" """

import numpy as np
from diffstar.defaults import T_TABLE_MIN
from diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.data_loaders.defaults import TransmissionCurve
from dsps.metallicity import umzr
from jax import random as jran

from ..scatter import DEFAULT_SCATTER_PARAMS
from ...param_utils import diffsky_param_wrapper as dpw
from ...param_utils import spspop_param_utils as spspu
from ...ssp_err_model import ssp_err_model
from .. import lc_phot_kern
from .. import mc_lightcone_halos as mclh


def test_multiband_lc_phot_kern():
    ran_key = jran.key(0)
    lgmp_min = 12.0
    z_min, z_max = 0.1, 0.5
    sky_area_degsq = 1.0

    args = (ran_key, lgmp_min, z_min, z_max, sky_area_degsq)

    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(*args)

    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()

    _res = retrieve_fake_fsps_data.load_fake_filter_transmission_curves()
    wave, u, g, r, i, z, y = _res

    tcurves = [TransmissionCurve(wave, x) for x in (u, g, r, i, z, y)]
    n_bands = len(tcurves)

    n_z_phot_table = 15
    z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)
    precomputed_ssp_mag_table = mclh.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table
    )
    n_z_table, n_bands, n_met, n_age = precomputed_ssp_mag_table.shape

    z_obs = lc_halopop["z_obs"]
    t_obs = lc_halopop["t_obs"]
    mah_params = lc_halopop["mah_params"]
    logmp0 = lc_halopop["logmp0"]
    t_0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    lgt0 = np.log10(t_0)

    t_table = np.linspace(T_TABLE_MIN, 10**lgt0, 100)

    wave_eff_table = lc_phot_kern.get_wave_eff_table(z_phot_table, tcurves)

    mzr_params = umzr.DEFAULT_MZR_PARAMS

    spspop_params = spspu.DEFAULT_SPSPOP_PARAMS
    scatter_params = DEFAULT_SCATTER_PARAMS
    ssp_err_pop_params = ssp_err_model.DEFAULT_SSPERR_PARAMS

    args = (
        ran_key,
        z_obs,
        t_obs,
        mah_params,
        logmp0,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        DEFAULT_DIFFSTARPOP_PARAMS,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
    )
    lc_phot = lc_phot_kern.multiband_lc_phot_kern(*args)
    for arr in lc_phot:
        assert np.all(np.isfinite(arr))


def _generate_lc_data():
    ran_key = jran.key(0)
    lgmp_min = 12.0
    z_min, z_max = 0.1, 0.5
    sky_area_degsq = 1.0

    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()

    _res = retrieve_fake_fsps_data.load_fake_filter_transmission_curves()
    wave, u, g, r, i, z, y = _res

    tcurves = [TransmissionCurve(wave, x) for x in (u, g, r, i, z, y)]

    n_z_phot_table = 15
    z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

    lc_data = lc_phot_kern.generate_lc_data(
        ran_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        ssp_data,
        DEFAULT_COSMOLOGY,
        tcurves,
        z_phot_table,
    )
    return lc_data


def test_generate_lc_data():
    lc_data = _generate_lc_data()
    assert np.all(np.isfinite(lc_data.logmp0))
    for x in lc_data.mah_params:
        assert np.all(np.isfinite(x))


def test_multiband_lc_phot_kern_u_param_arr():
    ran_key = jran.key(0)
    lc_data = _generate_lc_data()

    n_gals = lc_data.logmp0.size
    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape
    assert lc_data.z_phot_table.shape == (n_z_table,)
    assert lc_data.ssp_data.ssp_lgmet.shape == (n_met,)
    assert lc_data.ssp_data.ssp_lg_age_gyr.shape == (n_age,)

    u_param_collection = dpw.get_u_param_collection_from_param_collection(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    u_param_arr = dpw.unroll_u_param_collection_into_flat_array(*u_param_collection)

    lc_phot = lc_phot_kern.multiband_lc_phot_kern_u_param_arr(
        u_param_arr, ran_key, lc_data
    )
    for x in lc_phot:
        assert np.all(np.isfinite(x))

    assert lc_phot.obs_mags_bursty_ms.shape == (n_gals, n_bands)

    assert np.all(lc_phot.weights_bursty_ms >= 0)
    assert np.all(lc_phot.weights_bursty_ms <= 1)
    assert np.any(lc_phot.weights_bursty_ms > 0)
    assert np.any(lc_phot.weights_bursty_ms < 1)

    assert np.all(lc_phot.weights_smooth_ms >= 0)
    assert np.all(lc_phot.weights_smooth_ms <= 1)
    assert np.any(lc_phot.weights_smooth_ms > 0)
    assert np.any(lc_phot.weights_smooth_ms < 1)

    assert np.all(lc_phot.weights_q >= 0)
    assert np.all(lc_phot.weights_q <= 1)
    assert np.any(lc_phot.weights_q > 0)
    assert np.any(lc_phot.weights_q < 1)
