""" """

import numpy as np
from dsps.cosmology import flat_wcdm
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.data_loaders.defaults import TransmissionCurve
from jax import random as jran

from ...mass_functions import mc_hosts
from .. import mc_lightcone_halos as mclh


def test_mc_lightcone_host_halo_mass_function():
    """Enforce mc_lightcone_host_halo_mass_function produces consistent halo mass functions as
    the diffsky.mass_functions.mc_hosts function evaluated at the median redshift

    """
    lgmp_min = 12.0
    z_min, z_max = 0.1, 0.5
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
        vol_com = fsky * (vol_hi - vol_lo)

        lgmp_halopop_zmed = mc_hosts.mc_host_halos_singlez(
            ran_key, lgmp_min, z_med, vol_com
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


def test_mc_lightcone_diffstar_cens():
    """Enforce mc_lightcone_diffstar_cens returns reasonable results"""
    ran_key = jran.key(0)
    lgmp_min = 12.0
    sky_area_degsq = 1.0

    z_min, z_max = 0.1, 0.5
    z_min = z_max - 0.05
    args = (ran_key, lgmp_min, z_min, z_max, sky_area_degsq)
    cenpop = mclh.mc_lightcone_diffstar_cens(*args)
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
    cenpop = mclh.mc_lightcone_diffstar_stellar_ages_cens(*args)
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
    cenpop = mclh.mc_lightcone_diffstar_ssp_weights_cens(*args)
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
    cenpop = mclh.mc_lightcone_obs_mags_cens(*args)
    n_gals = cenpop["logsm_obs"].size

    assert np.all(np.isfinite(cenpop["obs_mags"]))
    assert cenpop["wave_eff"].shape == (n_gals, n_bands)
    assert np.all(np.isfinite(cenpop["wave_eff"]))
    assert np.all(cenpop["wave_eff"] > 100)
    assert np.all(cenpop["wave_eff"] < 1e5)
