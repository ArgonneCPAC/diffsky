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


def test_mc_diffsky_seds():
    ran_key = jran.key(0)
    lc_data, tcurves = tlcphk._generate_sobol_lc_data()
    n_gals = lc_data.logmp0.size

    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape
    assert lc_data.z_phot_table.shape == (n_z_table,)
    assert lc_data.ssp_data.ssp_lgmet.shape == (n_met,)
    assert lc_data.ssp_data.ssp_lg_age_gyr.shape == (n_age,)

    u_param_collection = dpw.get_u_param_collection_from_param_collection(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    u_param_arr = dpw.unroll_u_param_collection_into_flat_array(*u_param_collection)

    sed_info = mcsed.mc_diffsky_seds(u_param_arr, ran_key, lc_data)
    lc_phot_orig = lc_phot_kern.multiband_lc_phot_kern_u_param_arr(
        u_param_arr, ran_key, lc_data
    )
    msk_q = sed_info.mc_sfh_type == 0
    assert np.allclose(lc_phot_orig.obs_mags_q[msk_q], sed_info.obs_mags[msk_q])
    msk_smooth_ms = sed_info.mc_sfh_type == 1
    assert np.allclose(
        lc_phot_orig.obs_mags_smooth_ms[msk_smooth_ms], sed_info.obs_mags[msk_smooth_ms]
    )
    msk_bursty_ms = sed_info.mc_sfh_type == 2
    assert np.allclose(
        lc_phot_orig.obs_mags_bursty_ms[msk_bursty_ms], sed_info.obs_mags[msk_bursty_ms]
    )

    assert np.all(np.isfinite(sed_info.diffstar_params.ms_params))
    assert np.all(np.isfinite(sed_info.diffstar_params.q_params))

    assert np.all(np.isfinite(sed_info.burst_params.lgfburst))
    assert np.all(sed_info.burst_params.lgfburst >= LGFBURST_MIN)
    assert np.all(sed_info.burst_params.lgfburst <= LGFBURST_MAX)

    assert np.all(np.isfinite(sed_info.burst_params.lgyr_peak))
    assert np.all(sed_info.burst_params.lgyr_peak >= LGYR_PEAK_MIN)
    assert np.all(sed_info.burst_params.lgyr_peak <= LGYR_PEAK_MAX)

    assert np.all(np.isfinite(sed_info.burst_params.lgyr_max))

    assert np.all(np.isfinite(sed_info.logsm))
    assert np.all(sed_info.logsm > 5)
    assert np.all(sed_info.logsm < 13)

    # Enforce SSP weights sum to unity
    assert np.all(np.isfinite(sed_info.ssp_weights))
    assert sed_info.ssp_weights.shape == (n_gals, n_met, n_age)
    ssp_wtot = np.sum(sed_info.ssp_weights, axis=(1, 2))
    assert np.allclose(ssp_wtot, 1.0, rtol=1e-4)

    # Enforce agreement between precomputed vs exact magnitudes
    for iband in range(n_bands):
        args = (
            lc_data.ssp_data.ssp_wave,
            sed_info.rest_sed,
            tcurves[iband].wave,
            tcurves[iband].transmission,
            lc_data.z_obs,
            *DEFAULT_COSMOLOGY,
        )

        mags = calc_obs_mags_galpop(*args)
        mag_err = mags - sed_info.obs_mags[:, iband]
        assert np.mean(mag_err) < 0.05
        assert np.std(mag_err) < 0.1
