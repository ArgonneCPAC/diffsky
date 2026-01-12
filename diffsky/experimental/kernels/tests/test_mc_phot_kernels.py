""""""

import numpy as np
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.photometry import photometry_kernels as phk
from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from jax import random as jran
from jax import vmap

from ....param_utils import diffsky_param_wrapper as dpw
from ...tests import test_mc_lightcone_halos as tmclh
from ...tests import test_mc_phot
from .. import mc_phot_kernels as mcpk

_A = [None, 0, None, None, 0, *[None] * 4]
calc_obs_mags_galpop = vmap(phk.calc_obs_mag, in_axes=_A)


def test_sed_kern(num_halos=250):
    ran_key = jran.key(0)
    lc_data, tcurves = tmclh._get_weighted_lc_data_for_unit_testing(num_halos=num_halos)

    fb = 0.156
    ran_key, phot_key = jran.split(ran_key, 2)

    phot_kern_results, phot_randoms = mcpk._mc_phot_kern(
        phot_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpw.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [getattr(phot_kern_results, key) for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    sed_kern_results = mcpk._sed_kern(
        phot_randoms,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        *dpw.DEFAULT_PARAM_COLLECTION[1:],
        DEFAULT_COSMOLOGY,
        fb,
    )
    rest_sed_recomputed = sed_kern_results[0]

    # Enforce agreement between precomputed vs exact magnitudes
    n_bands = phot_kern_results.obs_mags.shape[1]
    for iband in range(n_bands):
        trans_iband = np.interp(
            lc_data.ssp_data.ssp_wave,
            tcurves[iband].wave,
            tcurves[iband].transmission,
        )
        args = (
            lc_data.ssp_data.ssp_wave,
            rest_sed_recomputed,
            lc_data.ssp_data.ssp_wave,
            trans_iband,
            lc_data.z_obs,
            *DEFAULT_COSMOLOGY,
        )

        mags = calc_obs_mags_galpop(*args)
        assert np.allclose(mags, phot_kern_results.obs_mags[:, iband], rtol=0.01)


def test_mc_dbk_kern(num_halos=50):
    ran_key = jran.key(0)
    lc_data, tcurves = tmclh._get_weighted_lc_data_for_unit_testing(num_halos=num_halos)

    fb = 0.156
    ran_key, phot_key = jran.split(ran_key, 2)

    phot_kern_results, phot_randoms = mcpk._mc_phot_kern(
        phot_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpw.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )

    test_mc_phot.check_phot_kern_results(phot_kern_results)

    ran_key, dbk_key = jran.split(ran_key, 2)
    burst_params = DEFAULT_BURST_PARAMS._replace(
        lgfburst=phot_kern_results.lgfburst,
        lgyr_peak=phot_kern_results.lgyr_peak,
        lgyr_max=phot_kern_results.lgyr_max,
    )
    args = (
        lc_data.t_obs,
        lc_data.ssp_data,
        phot_kern_results.t_table,
        phot_kern_results.sfh_table,
        burst_params,
        phot_kern_results.lgmet_weights,
        dbk_key,
    )
    dbk_randoms, dbk_weights, disk_bulge_history = mcpk._mc_dbk_kern(*args)
    assert np.all(np.isfinite(dbk_weights.ssp_weights_bulge))
    assert np.all(np.isfinite(dbk_weights.ssp_weights_disk))
    assert np.all(np.isfinite(dbk_weights.ssp_weights_knots))

    assert np.allclose(
        np.sum(dbk_weights.ssp_weights_bulge, axis=(1, 2)), 1.0, rtol=1e-4
    )
    assert np.allclose(
        np.sum(dbk_weights.ssp_weights_disk, axis=(1, 2)), 1.0, rtol=1e-4
    )
    assert np.allclose(
        np.sum(dbk_weights.ssp_weights_knots, axis=(1, 2)), 1.0, rtol=1e-4
    )

    assert np.all(dbk_weights.mstar_bulge > 0)
    assert np.all(dbk_weights.mstar_disk > 0)
    assert np.all(dbk_weights.mstar_knots > 0)

    args = (
        phot_kern_results.ssp_photflux_table,
        dbk_weights,
        phot_kern_results.dust_frac_trans,
        phot_kern_results.frac_ssp_errors,
    )
    _res = mcpk._get_dbk_phot_from_dbk_weights(*args)
    obs_mags_bulge, obs_mags_disk, obs_mags_knots = _res

    np.all(phot_kern_results.logsm_obs > np.log10(dbk_weights.mstar_bulge.flatten()))
    np.all(phot_kern_results.logsm_obs > np.log10(dbk_weights.mstar_disk.flatten()))
    np.all(phot_kern_results.logsm_obs > np.log10(dbk_weights.mstar_knots.flatten()))

    assert np.all(np.isfinite(obs_mags_bulge))
    assert np.all(np.isfinite(obs_mags_disk))
    assert np.all(np.isfinite(obs_mags_knots))

    assert not np.allclose(phot_kern_results.obs_mags, obs_mags_bulge, rtol=1e-4)
    assert np.all(phot_kern_results.obs_mags <= obs_mags_bulge)

    assert not np.allclose(phot_kern_results.obs_mags, obs_mags_disk, rtol=1e-4)
    assert np.all(phot_kern_results.obs_mags <= obs_mags_disk)

    assert not np.allclose(phot_kern_results.obs_mags, obs_mags_knots, rtol=1e-4)
    assert np.all(phot_kern_results.obs_mags <= obs_mags_knots)

    a = 10 ** (-0.4 * obs_mags_bulge)
    b = 10 ** (-0.4 * obs_mags_disk)
    c = 10 ** (-0.4 * obs_mags_knots)
    mtot = -2.5 * np.log10(a + b + c)

    magdiff = mtot - phot_kern_results.obs_mags
    assert np.all(np.abs(magdiff) < 0.1)

    mean_magdiff = np.mean(magdiff, axis=0)  # shape = (n_bands,)
    assert np.allclose(mean_magdiff, 0.0, atol=0.01)

    std_magdiff = np.std(magdiff, axis=0)
    assert np.all(std_magdiff < 0.01)
