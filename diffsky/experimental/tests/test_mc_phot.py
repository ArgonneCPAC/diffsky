""""""

from collections import namedtuple

import numpy as np
from diffstar.diffstarpop.kernels.params import (
    DiffstarPop_Params_Diffstarpopfits_mgash as sfh_models,
)
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.photometry import photometry_kernels as phk
from jax import random as jran
from jax import vmap

from .. import mc_phot
from . import test_mc_lightcone_halos as tmclh

_A = [None, 0, None, None, 0, *[None] * 4]
calc_obs_mags_galpop = vmap(phk.calc_obs_mag, in_axes=_A)


def test_mc_lc_phot_changes_with_diffstarpop(num_halos=50):
    ran_key = jran.key(0)
    lc_data, tcurves = tmclh._get_weighted_lc_data_for_unit_testing(num_halos=num_halos)
    phot_kern_results = mc_phot.mc_lc_phot(
        ran_key, lc_data, diffstarpop_params=sfh_models["tng"]
    )
    phot_kern_results2 = mc_phot.mc_lc_phot(
        ran_key, lc_data, diffstarpop_params=sfh_models["smdpl_dr1"]
    )
    assert not np.allclose(
        phot_kern_results["obs_mags"], phot_kern_results2["obs_mags"], atol=0.1
    )

    keys = list(phot_kern_results.keys())
    phot_kern_results = namedtuple("Results", keys)(**phot_kern_results)
    phot_kern_results2 = namedtuple("Results", keys)(**phot_kern_results2)
    check_phot_kern_results(phot_kern_results)
    check_phot_kern_results(phot_kern_results2)


def test_mc_lc_sed_is_consistent_with_mc_lc_phot(num_halos=50):
    ran_key = jran.key(0)
    lc_data, tcurves = tmclh._get_weighted_lc_data_for_unit_testing(num_halos=num_halos)
    phot_kern_results = mc_phot.mc_lc_phot(ran_key, lc_data)
    sed_kern_results = mc_phot.mc_lc_sed(ran_key, lc_data)

    phot_kern_results = namedtuple("Results", list(phot_kern_results.keys()))(
        **phot_kern_results
    )
    rest_sed_recomputed = sed_kern_results["rest_sed"]

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


def test_mc_lc_dbk_phot(num_halos=50):
    ran_key = jran.key(0)
    lc_data, tcurves = tmclh._get_weighted_lc_data_for_unit_testing(num_halos=num_halos)
    dbk_phot_info = mc_phot.mc_lc_dbk_phot(ran_key, lc_data)

    np.all(dbk_phot_info["logsm_obs"] > np.log10(dbk_phot_info["mstar_bulge"]))
    np.all(dbk_phot_info["logsm_obs"] > np.log10(dbk_phot_info["mstar_disk"]))
    np.all(dbk_phot_info["logsm_obs"] > np.log10(dbk_phot_info["mstar_knots"]))

    assert not np.allclose(
        dbk_phot_info["obs_mags"], dbk_phot_info["obs_mags_bulge"], rtol=1e-4
    )
    assert np.all(dbk_phot_info["obs_mags"] <= dbk_phot_info["obs_mags_bulge"])

    assert not np.allclose(
        dbk_phot_info["obs_mags"], dbk_phot_info["obs_mags_disk"], rtol=1e-4
    )
    assert np.all(dbk_phot_info["obs_mags"] <= dbk_phot_info["obs_mags_disk"])

    assert not np.allclose(
        dbk_phot_info["obs_mags"], dbk_phot_info["obs_mags_knots"], rtol=1e-4
    )
    assert np.all(dbk_phot_info["obs_mags"] <= dbk_phot_info["obs_mags_knots"])

    a = 10 ** (-0.4 * dbk_phot_info["obs_mags_bulge"])
    b = 10 ** (-0.4 * dbk_phot_info["obs_mags_disk"])
    c = 10 ** (-0.4 * dbk_phot_info["obs_mags_knots"])
    mtot = -2.5 * np.log10(a + b + c)

    magdiff = mtot - dbk_phot_info["obs_mags"]
    assert np.all(np.abs(magdiff) < 0.1)

    mean_magdiff = np.mean(magdiff, axis=0)  # shape = (n_bands,)
    assert np.allclose(mean_magdiff, 0.0, atol=0.01)

    std_magdiff = np.std(magdiff, axis=0)
    assert np.all(std_magdiff < 0.01)


def check_phot_kern_results(phot_kern_results):
    assert np.all(np.isfinite(phot_kern_results.obs_mags))
    assert np.all(phot_kern_results.lgfburst[phot_kern_results.mc_sfh_type < 2] < -7)

    assert np.allclose(
        np.sum(phot_kern_results.ssp_weights, axis=(1, 2)), 1.0, rtol=1e-4
    )
    assert np.all(phot_kern_results.frac_ssp_errors > 0)
    assert np.all(phot_kern_results.frac_ssp_errors < 5)


def test_mc_lc_dbk_sed(num_halos=50):
    ran_key = jran.key(0)
    lc_data, tcurves = tmclh._get_weighted_lc_data_for_unit_testing(num_halos=num_halos)
    dbk_sed_info = mc_phot.mc_lc_dbk_sed(ran_key, lc_data)
    assert np.all(np.isfinite(dbk_sed_info["rest_sed_bulge"]))
    assert np.all(np.isfinite(dbk_sed_info["rest_sed_disk"]))
    assert np.all(np.isfinite(dbk_sed_info["rest_sed_knots"]))


def test_unweighted_mc_lc_dbk_sed():
    ran_key = jran.key(0)
    lc_data, tcurves = tmclh._get_unweighted_lc_data_for_unit_testing()
    dbk_sed_info = mc_phot.mc_lc_dbk_sed(ran_key, lc_data)
    assert np.all(np.isfinite(dbk_sed_info["rest_sed_bulge"]))
    assert np.all(np.isfinite(dbk_sed_info["rest_sed_disk"]))
    assert np.all(np.isfinite(dbk_sed_info["rest_sed_knots"]))
