""" """

import numpy as np
from dsps.cosmology import flat_wcdm
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.data_loaders.defaults import TransmissionCurve
from jax import random as jran

from ...mass_functions import mc_hosts
from ...mass_functions.fitting_utils.calibrations import hacc_core_shmf_params as hcshmf
from .. import mc_lightcone_halos as mclh


def test_mc_lightcone_host_halo_mass_function():
    """Enforce mc_lightcone_host_halo_mass_function produces consistent halo mass functions as
    the diffsky.mass_functions.mc_hosts function evaluated at the median redshift

    """
    lgmp_min = 12.0
    z_min, z_max = 0.4, 0.5
    sky_area_degsq = 10.0

    n_tests = 5
    ran_keys = jran.split(jran.key(0), n_tests)
    for ran_key in ran_keys:
        args = (ran_key, lgmp_min, z_min, z_max, sky_area_degsq)

        redshifts_galpop, logmp_halopop = mclh.mc_lightcone_host_halo_mass_function(
            *args
        )
        assert np.all(np.isfinite(redshifts_galpop))
        assert np.all(np.isfinite(logmp_halopop))
        assert logmp_halopop.shape == redshifts_galpop.shape
        assert np.all(redshifts_galpop >= z_min)
        assert np.all(redshifts_galpop <= z_max)
        assert np.all(logmp_halopop > lgmp_min)

        z_med = np.median(redshifts_galpop)

        cosmo_params = flat_wcdm.PLANCK15
        vol_lo = (
            (4 / 3)
            * np.pi
            * flat_wcdm.comoving_distance_to_z(z_min, *cosmo_params) ** 3
        )
        vol_hi = (
            (4 / 3)
            * np.pi
            * flat_wcdm.comoving_distance_to_z(z_max, *cosmo_params) ** 3
        )
        fsky = sky_area_degsq / mclh.FULL_SKY_AREA
        vol_com_mpc = fsky * (vol_hi - vol_lo)

        lgmp_halopop_zmed = mc_hosts.mc_host_halos_singlez(
            ran_key, lgmp_min, z_med, vol_com_mpc
        )

        n_lightcone, n_snapshot = redshifts_galpop.size, lgmp_halopop_zmed.size
        fracdiff = (n_lightcone - n_snapshot) / n_snapshot
        assert np.abs(fracdiff) < 0.05

        lgmp_hist_lc, lgmp_bins = np.histogram(logmp_halopop, bins=50)
        lgmp_hist_zmed, lgmp_bins = np.histogram(lgmp_halopop_zmed, bins=lgmp_bins)
        msk_counts = lgmp_hist_zmed > 500
        fracdiff = (
            lgmp_hist_lc[msk_counts] - lgmp_hist_zmed[msk_counts]
        ) / lgmp_hist_zmed[msk_counts]
        assert np.all(np.abs(fracdiff) < 0.1)


def test_mc_lightcone_host_halo_mass_function_lgmp_max_feature():
    """Using lgmp_max gives the same halo counts as when not using it and masking"""
    ran_key = jran.key(0)
    lgmp_min = 12.0
    z_min, z_max = 0.45, 0.5
    sky_area_degsq = 200.0

    lgmp_max_test = 13.0

    n_tests = 10
    for __ in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        args = (test_key, lgmp_min, z_min, z_max, sky_area_degsq)

        z_halopop, logmp_halopop = mclh.mc_lightcone_host_halo_mass_function(*args)
        z_halopop2, logmp_halopop2 = mclh.mc_lightcone_host_halo_mass_function(
            *args, lgmp_max=lgmp_max_test
        )
        assert z_halopop.size > z_halopop2.size
        assert z_halopop2.size == logmp_halopop2.size

        n1 = np.sum(logmp_halopop < lgmp_max_test)
        n2 = logmp_halopop2.size
        assert np.allclose(n1, n2, rtol=0.02)


def test_nhalo_weighted_lc_grid():
    """should get within 10% of Nhalos of a Monte Carlo generated lightcone"""
    ran_key = jran.key(0)
    n_tests = 10

    sky_area_degsq = 15.0

    for itest in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)

        lgm_key, z_key, lc_key = jran.split(test_key, 3)
        lgmp_min = jran.uniform(lgm_key, minval=11, maxval=12, shape=())
        z_min = jran.uniform(z_key, minval=0.1, maxval=1.0, shape=())
        z_max = z_min + 1

        n_z, n_m = 150, 250
        lgmp_grid = np.linspace(lgmp_min, 15, n_m)
        z_grid = np.linspace(z_min, z_max, n_z)

        nhalo_weighted_lc_grid = mclh.get_nhalo_weighted_lc_grid(
            lgmp_grid,
            z_grid,
            sky_area_degsq,
            hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
            cosmo_params=flat_wcdm.PLANCK15,
        )
        assert nhalo_weighted_lc_grid.shape == (n_z, n_m)

        args = (lc_key, lgmp_min, z_min, z_max, sky_area_degsq)
        z_halopop, logmp_halopop = mclh.mc_lightcone_host_halo_mass_function(*args)

        assert np.allclose(logmp_halopop.size, nhalo_weighted_lc_grid.sum(), rtol=0.1)


def test_get_weighted_lightcone_grid_host_halo_diffmah():
    ran_key = jran.key(0)
    n_tests = 10

    sky_area_degsq = 15.0

    for itest in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)

        lgm_key, z_key, lc_key = jran.split(test_key, 3)
        lgmp_min = jran.uniform(lgm_key, minval=11, maxval=12, shape=())
        z_min = jran.uniform(z_key, minval=0.1, maxval=1.0, shape=())
        z_max = z_min + 1

        n_z, n_m = 150, 250
        lgmp_grid = np.linspace(lgmp_min, 15, n_m)
        z_grid = np.linspace(z_min, z_max, n_z)

        cenpop = mclh.get_weighted_lightcone_grid_host_halo_diffmah(
            ran_key, lgmp_grid, z_grid, sky_area_degsq
        )
        assert "nhalos" in cenpop.keys()


def test_mc_lightcone_host_halo_diffmah():
    """Enforce mc_lightcone_host_halo_diffmah returns reasonable results"""
    ran_key = jran.key(0)
    lgmp_min = 12.0
    sky_area_degsq = 1.0

    n_tests = 5
    z_max_arr = np.linspace(0.2, 2.5, n_tests)
    for z_max in z_max_arr:
        test_key, ran_key = jran.split(ran_key, 2)
        z_min = z_max - 0.05
        args = (test_key, lgmp_min, z_min, z_max, sky_area_degsq)

        cenpop = mclh.mc_lightcone_host_halo_diffmah(*args)
        n_gals = cenpop["z_obs"].size
        assert cenpop["logmp_obs"].size == cenpop["logmp0"].size == n_gals
        assert np.all(np.isfinite(cenpop["z_obs"]))

        assert np.all(cenpop["z_obs"] >= z_min)
        assert np.all(cenpop["z_obs"] <= z_max)

        # Some halos with logmp_obs<lgmp_min is ok,
        # but too many indicates an issue with DiffmahPop replicating logmp_obs
        assert np.mean(cenpop["logmp_obs"] < lgmp_min) < 0.2, f"z_min={z_min:.2f}"


def test_mc_lightcone_host_halo_diffmah_alt_mf_params():
    """Enforce mc_lightcone_host_halo_diffmah returns reasonable results when passed
    alternative halo mass function parameters"""
    ran_key = jran.key(0)
    lgmp_min = 12.0
    sky_area_degsq = 1.0

    n_tests = 5
    z_max_arr = np.linspace(0.2, 2.5, n_tests)
    for z_max in z_max_arr:
        test_key, ran_key = jran.split(ran_key, 2)
        z_min = z_max - 0.05
        args = (test_key, lgmp_min, z_min, z_max, sky_area_degsq)

        cenpop = mclh.mc_lightcone_host_halo_diffmah(
            *args, hmf_params=hcshmf.HMF_PARAMS
        )
        n_gals = cenpop["z_obs"].size
        assert cenpop["logmp_obs"].size == cenpop["logmp0"].size == n_gals
        assert np.all(np.isfinite(cenpop["z_obs"]))

        assert np.all(cenpop["z_obs"] >= z_min)
        assert np.all(cenpop["z_obs"] <= z_max)

        # Some halos with logmp_obs<lgmp_min is ok,
        # but too many indicates an issue with DiffmahPop replicating logmp_obs
        assert np.mean(cenpop["logmp_obs"] < lgmp_min) < 0.2, f"z_min={z_min:.2f}"


def test_mc_lightcone_diffstar_cens():
    """Enforce mc_lightcone_diffstar_cens returns reasonable results"""
    ran_key = jran.key(0)
    lgmp_min = 12.0
    sky_area_degsq = 1.0

    z_min, z_max = 0.1, 0.5
    z_min = z_max - 0.05
    args = (ran_key, lgmp_min, z_min, z_max, sky_area_degsq)
    cenpop = mclh.mc_lightcone_diffstar_cens(*args, return_internal_quantities=True)
    assert np.all(np.isfinite(cenpop["logsm_obs"]))
    assert np.all(np.isfinite(cenpop["logssfr_obs"]))
    assert cenpop["logsm_obs"].min() > 4

    assert np.all(np.isfinite(cenpop["diffstarpop_data"]["sfh_params_ms"].ms_params))
    assert np.all(np.isfinite(cenpop["diffstarpop_data"]["sfh_params_ms"].q_params))
    assert np.all(np.isfinite(cenpop["diffstarpop_data"]["sfh_params_q"].ms_params))
    assert np.all(np.isfinite(cenpop["diffstarpop_data"]["sfh_params_q"].q_params))

    assert np.all(np.isfinite(cenpop["diffstarpop_data"]["sfh_ms"]))
    assert np.all(np.isfinite(cenpop["diffstarpop_data"]["sfh_q"]))
    assert np.all(cenpop["diffstarpop_data"]["frac_q"] >= 0)
    assert np.all(cenpop["diffstarpop_data"]["frac_q"] <= 1)


def test_mc_lightcone_diffstar_stellar_ages_cens():
    """Enforce mc_lightcone_diffstar_stellar_ages_cens returns reasonable results"""
    ran_key = jran.key(0)
    lgmp_min = 12.0
    sky_area_degsq = 1.0

    z_min, z_max = 0.1, 0.5
    z_min = z_max - 0.05

    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()

    args = (ran_key, lgmp_min, z_min, z_max, sky_area_degsq, ssp_data)
    cenpop = mclh.mc_lightcone_diffstar_stellar_ages_cens(
        *args, return_internal_quantities=True
    )
    assert np.all(np.isfinite(cenpop["logsm_obs"]))
    assert np.all(np.isfinite(cenpop["logssfr_obs"]))
    assert cenpop["logsm_obs"].min() > 4

    assert np.all(np.isfinite(cenpop["age_weights"]))
    assert np.allclose(1.0, np.sum(cenpop["age_weights"], axis=1), rtol=1e-3)


def test_mc_lightcone_diffstar_ssp_weights_cens():
    ran_key = jran.key(0)
    lgmp_min = 12.0
    sky_area_degsq = 1.0

    z_min, z_max = 0.1, 0.5
    z_min = z_max - 0.05

    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()

    args = (ran_key, lgmp_min, z_min, z_max, sky_area_degsq, ssp_data)
    cenpop = mclh.mc_lightcone_diffstar_ssp_weights_cens(
        *args, return_internal_quantities=True
    )
    assert np.all(np.isfinite(cenpop["logsm_obs"]))
    assert np.all(np.isfinite(cenpop["logssfr_obs"]))
    assert cenpop["logsm_obs"].min() > 4

    assert np.all(np.isfinite(cenpop["age_weights"]))
    assert np.allclose(1.0, np.sum(cenpop["age_weights"], axis=1), rtol=1e-3)

    assert np.all(np.isfinite(cenpop["lgmet_weights"]))
    assert np.allclose(1.0, np.sum(cenpop["lgmet_weights"], axis=1), rtol=1e-3)

    assert np.all(np.isfinite(cenpop["ssp_weights"]))
    assert np.allclose(1.0, np.sum(cenpop["ssp_weights"], axis=(1, 2)), rtol=1e-3)


def test_mc_lightcone_obs_mags_cens():
    ran_key = jran.key(0)
    lgmp_min = 12.0
    sky_area_degsq = 1.0

    z_min, z_max = 0.1, 0.5
    z_min = z_max - 0.05

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

    args = (
        ran_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        ssp_data,
        tcurves,
        precomputed_ssp_mag_table,
        z_phot_table,
    )
    cenpop = mclh.mc_lightcone_obs_mags_cens(*args, return_internal_quantities=True)
    n_gals = cenpop["logsm_obs"].size

    assert np.all(np.isfinite(cenpop["obs_mags"]))
    assert cenpop["wave_eff"].shape == (n_gals, n_bands)
    assert np.all(np.isfinite(cenpop["wave_eff"]))
    assert np.all(cenpop["wave_eff"] > 100)
    assert np.all(cenpop["wave_eff"] < 1e5)

    assert cenpop["obs_mags_nodust_noerr"].shape == (n_gals, n_bands)
    assert cenpop["obs_mags"].shape == (n_gals, n_bands)
    assert np.all(cenpop["obs_mags_noerr"] >= cenpop["obs_mags_nodust_noerr"])

    assert np.all(cenpop["ftrans"] >= 0)
    assert np.all(cenpop["ftrans"] <= 1)
    assert np.any(cenpop["ftrans"] > 0)
    assert np.any(cenpop["ftrans"] < 1)
