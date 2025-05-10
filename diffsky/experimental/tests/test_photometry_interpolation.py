"""
"""

import numpy as np
from ..photometry_interpolation import get_interpolated_photometry
from ..photometry_interpolation import calc_ssp_weights_sfh_table_lognormal_mdf_vmap
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z


def test_vmap_weights():
    n_met, n_age = 12, 50
    ssp_lgmet = np.linspace(-4, 0.5, n_met)
    ssp_lg_age = np.linspace(5.5, 10.5, n_age) - 9

    n_t_table = 80
    gal_t_table = np.linspace(0.5, 13.8, n_t_table)

    n_gals = 500
    gal_z_obs = np.random.uniform(0.05, 2.95, n_gals)
    gal_t_obs = age_at_z(gal_z_obs, *DEFAULT_COSMOLOGY)

    gal_sfr_table = np.random.uniform(0, 1, size=(n_gals, n_t_table))
    gal_lgmet_obs = np.random.uniform(-2, 0, n_gals)
    gal_lgmet_scatter = 0.2
    _res = calc_ssp_weights_sfh_table_lognormal_mdf_vmap(
        gal_t_table,
        gal_sfr_table,
        gal_lgmet_obs,
        gal_lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age,
        gal_t_obs,
    )
    gal_weights, gal_lgmet_weights, gal_age_weights = _res
    assert np.all(np.isfinite(gal_weights))

    epsilon = 1e-5
    assert np.all(gal_weights >= 0 - epsilon)
    assert np.all(gal_weights <= 1 + epsilon)
    assert gal_weights.shape == (n_gals, n_met, n_age)

    assert np.allclose(np.sum(gal_weights, axis=(1, 2)), 1.0, rtol=1e-3)


def test_get_interpolated_photometry_nodust():
    n_z_table_ssp = 20
    ssp_z_table = np.linspace(0.02, 3, n_z_table_ssp)

    n_met, n_age = 12, 50
    n_rest_filters, n_obs_filters = 3, 6
    ssp_restmag_table = np.random.uniform(0, 1, size=(n_met, n_age, n_rest_filters))
    ssp_obsmag_table = np.random.uniform(
        0, 1, size=(n_z_table_ssp, n_met, n_age, n_obs_filters)
    )
    ssp_lgmet = np.linspace(-4, 0.5, n_met)
    ssp_lg_age = np.linspace(5.5, 10.5, n_age) - 9

    n_t_table = 80
    gal_t_table = np.linspace(0.5, 13.8, n_t_table)

    n_gals = 500
    gal_z_obs = np.random.uniform(0.05, 2.95, n_gals)
    gal_logsm_obs = np.random.uniform(8, 12, n_gals)
    gal_sfr_table = np.random.uniform(0, 1, size=(n_gals, n_t_table))
    gal_lgmet_obs = np.random.uniform(-2, 0, n_gals)
    gal_lgmet_scatter = 0.2

    res = get_interpolated_photometry(
        ssp_z_table,
        ssp_restmag_table,
        ssp_obsmag_table,
        ssp_lgmet,
        ssp_lg_age,
        gal_t_table,
        gal_z_obs,
        gal_logsm_obs,
        gal_sfr_table,
        gal_lgmet_obs,
        gal_lgmet_scatter,
        DEFAULT_COSMOLOGY,
    )

    gal_obsmags, gal_restmags, gal_obsmags_nodust, gal_restmags_nodust = res

    assert gal_obsmags.shape == (n_gals, n_obs_filters)
    assert gal_restmags.shape == (n_gals, n_rest_filters)
    assert gal_obsmags_nodust.shape == (n_gals, n_obs_filters)
    assert gal_restmags_nodust.shape == (n_gals, n_rest_filters)

    for mags in res:
        assert np.all(np.isfinite(mags))

    assert np.allclose(gal_obsmags, gal_obsmags_nodust)
    assert np.allclose(gal_restmags, gal_restmags_nodust)


def test_get_interpolated_photometry_with_dust():
    n_z_table_ssp = 20
    ssp_z_table = np.linspace(0.02, 3, n_z_table_ssp)

    n_met, n_age = 12, 50
    n_rest_filters, n_obs_filters = 3, 6
    ssp_restmag_table = np.random.uniform(0, 1, size=(n_met, n_age, n_rest_filters))
    ssp_obsmag_table = np.random.uniform(
        0, 1, size=(n_z_table_ssp, n_met, n_age, n_obs_filters)
    )
    ssp_lgmet = np.linspace(-4, 0.5, n_met)
    ssp_lg_age = np.linspace(5.5, 10.5, n_age) - 9

    n_t_table = 80
    gal_t_table = np.linspace(0.5, 13.8, n_t_table)

    n_gals = 500
    gal_z_obs = np.random.uniform(0.05, 2.95, n_gals)
    gal_logsm_obs = np.random.uniform(8, 12, n_gals)
    gal_sfr_table = np.random.uniform(0, 1, size=(n_gals, n_t_table))
    gal_lgmet_obs = np.random.uniform(-2, 0, n_gals)
    gal_lgmet_scatter = 0.2

    dust_trans_factors_obs = np.random.uniform(0, 1, size=(n_gals, n_obs_filters))
    dust_trans_factors_rest = np.random.uniform(0, 1, size=(n_gals, n_rest_filters))

    res = get_interpolated_photometry(
        ssp_z_table,
        ssp_restmag_table,
        ssp_obsmag_table,
        ssp_lgmet,
        ssp_lg_age,
        gal_t_table,
        gal_z_obs,
        gal_logsm_obs,
        gal_sfr_table,
        gal_lgmet_obs,
        gal_lgmet_scatter,
        DEFAULT_COSMOLOGY,
        dust_trans_factors_obs,
        dust_trans_factors_rest,
    )

    gal_obsmags, gal_restmags, gal_obsmags_nodust, gal_restmags_nodust = res

    assert gal_obsmags.shape == (n_gals, n_obs_filters)
    assert gal_restmags.shape == (n_gals, n_rest_filters)
    assert gal_obsmags_nodust.shape == (n_gals, n_obs_filters)
    assert gal_restmags_nodust.shape == (n_gals, n_rest_filters)

    for mags in res:
        assert np.all(np.isfinite(mags))

    assert not np.allclose(gal_obsmags, gal_obsmags_nodust)
    assert not np.allclose(gal_restmags, gal_restmags_nodust)

    assert np.all(gal_obsmags >= gal_obsmags_nodust)
    assert np.all(gal_restmags >= gal_restmags_nodust)
