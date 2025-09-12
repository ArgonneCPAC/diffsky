""" """

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.photometry import photometry_kernels as phk
from dsps.sfh.diffburst import LGFBURST_MAX, LGFBURST_MIN, LGYR_PEAK_MAX, LGYR_PEAK_MIN
from jax import random as jran
from jax import vmap

from ...param_utils import diffsky_param_wrapper as dpw
from .. import lc_phot_kern
from .. import mc_diffsky_seds as mcsed
from . import test_lc_phot_kern as tlcphk

_A = [None, 0, None, None, 0, *[None] * 4]
calc_obs_mags_galpop = vmap(phk.calc_obs_mag, in_axes=_A)


def test_mc_weighted_diffsky_lightcone():
    ran_key = jran.key(0)
    lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing()
    sed_info = mcsed.mc_weighted_diffsky_lightcone(ran_key, lc_data)

    _check_sed_info(sed_info, lc_data, tcurves)


def test_mc_diffsky_seds_flat_u_params():
    ran_key = jran.key(0)
    lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing()

    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape
    assert lc_data.z_phot_table.shape == (n_z_table,)
    assert lc_data.ssp_data.ssp_lgmet.shape == (n_met,)
    assert lc_data.ssp_data.ssp_lg_age_gyr.shape == (n_age,)

    u_param_collection = dpw.get_u_param_collection_from_param_collection(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    u_param_arr = dpw.unroll_u_param_collection_into_flat_array(*u_param_collection)

    sed_info = mcsed._mc_diffsky_seds_flat_u_params(
        u_param_arr, ran_key, lc_data, DEFAULT_COSMOLOGY
    )
    lc_phot_orig = lc_phot_kern.multiband_lc_phot_kern_u_param_arr(
        u_param_arr, ran_key, lc_data
    )
    msk_q = sed_info["mc_sfh_type"] == 0
    assert np.allclose(lc_phot_orig.obs_mags_q[msk_q], sed_info["obs_mags"][msk_q])
    msk_smooth_ms = sed_info["mc_sfh_type"] == 1
    assert np.allclose(
        lc_phot_orig.obs_mags_smooth_ms[msk_smooth_ms],
        sed_info["obs_mags"][msk_smooth_ms],
    )
    msk_bursty_ms = sed_info["mc_sfh_type"] == 2
    assert np.allclose(
        lc_phot_orig.obs_mags_bursty_ms[msk_bursty_ms],
        sed_info["obs_mags"][msk_bursty_ms],
    )

    _check_sed_info(sed_info, lc_data, tcurves)


def _check_sed_info(sed_info, lc_data, tcurves):
    n_gals = lc_data.logmp0.size
    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape

    assert np.all(np.isfinite(sed_info["diffstar_params"].ms_params))
    assert np.all(np.isfinite(sed_info["diffstar_params"].q_params))

    assert np.all(np.isfinite(sed_info["burst_params"].lgfburst))
    assert np.all(sed_info["burst_params"].lgfburst >= LGFBURST_MIN)
    assert np.all(sed_info["burst_params"].lgfburst <= LGFBURST_MAX)

    assert np.all(np.isfinite(sed_info["burst_params"].lgyr_peak))
    assert np.all(sed_info["burst_params"].lgyr_peak >= LGYR_PEAK_MIN)
    assert np.all(sed_info["burst_params"].lgyr_peak <= LGYR_PEAK_MAX)

    assert np.all(np.isfinite(sed_info["burst_params"].lgyr_max))

    assert sed_info["logmp_obs"].shape == (n_gals,)
    assert np.all(np.isfinite(sed_info["logmp_obs"]))
    assert np.all(sed_info["logmp_obs"] > 5)
    assert np.all(sed_info["logmp_obs"] <= 18)

    assert sed_info["logsm_obs"].shape == (n_gals,)
    assert np.all(np.isfinite(sed_info["logsm_obs"]))
    assert np.all(sed_info["logsm_obs"] > 5)
    assert np.all(sed_info["logsm_obs"] < 13)

    assert sed_info["logssfr_obs"].shape == (n_gals,)
    assert np.all(np.isfinite(sed_info["logssfr_obs"]))
    assert np.all(sed_info["logssfr_obs"] > -100)
    assert np.all(sed_info["logssfr_obs"] < -5)

    assert sed_info["sfh_table"].shape[0] == n_gals
    assert np.all(np.isfinite(sed_info["sfh_table"]))
    assert np.all(sed_info["sfh_table"] > 0)

    # Enforce SSP weights sum to unity
    assert np.all(np.isfinite(sed_info["ssp_weights"]))
    assert sed_info["ssp_weights"].shape == (n_gals, n_met, n_age)
    ssp_wtot = np.sum(sed_info["ssp_weights"], axis=(1, 2))
    assert np.allclose(ssp_wtot, 1.0, rtol=1e-4)

    # Enforce agreement between precomputed vs exact magnitudes
    for iband in range(n_bands):
        args = (
            lc_data.ssp_data.ssp_wave,
            sed_info["rest_sed"],
            tcurves[iband].wave,
            tcurves[iband].transmission,
            lc_data.z_obs,
            *DEFAULT_COSMOLOGY,
        )

        mags = calc_obs_mags_galpop(*args)
        mag_err = mags - sed_info["obs_mags"][:, iband]
        assert np.mean(mag_err) < 0.05
        assert np.std(mag_err) < 0.1

    assert np.all(np.isfinite(sed_info["dust_params"].av))
    assert np.all(np.isfinite(sed_info["dust_params"].delta))
    assert np.all(np.isfinite(sed_info["dust_params"].funo))

    assert sed_info["dust_params"].av.shape == (n_gals, n_age)
    assert sed_info["dust_params"].delta.shape == (n_gals,)
    assert sed_info["dust_params"].funo.shape == (n_gals,)

    # Enforce broadly reasonable values of Av
    assert np.all(sed_info["dust_params"].av > 0)
    assert np.all(sed_info["dust_params"].av < 5)

    # Enforce that av strictly decreases with stellar age
    assert np.all(np.diff(sed_info["dust_params"].av, axis=1) <= 0)
    assert np.any(np.diff(sed_info["dust_params"].av, axis=1) < 0)

    # Enforce broadly reasonable values of delta
    assert np.all(sed_info["dust_params"].delta > -2)
    assert np.all(sed_info["dust_params"].delta < 2)

    # Enforce physically sensible values of funo
    assert np.all(sed_info["dust_params"].funo > 0)
    assert np.all(sed_info["dust_params"].funo < 1)
