"""
"""
from jax import random as jran
import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY

from ..photpop import get_obs_photometry_singlez
from ..boris_dust import DEFAULT_PARAMS as DEFAULT_BORIS_PARAMS
from ..lgfburstpop import DEFAULT_LGFBURST_U_PARAMS
from ..burstshapepop import DEFAULT_BURSTSHAPE_U_PARAMS
from ..lgavpop import DEFAULT_LGAV_U_PARAMS
from ..dust_deltapop import DEFAULT_DUST_DELTA_U_PARAMS


def test_newphotpop_evaluates():
    n_met, n_age = 20, 50
    ssp_lgmet = np.linspace(-4, 0, n_met)
    ssp_lg_age = np.linspace(5.5, 10.5, n_age) - 9

    n_t = 100
    gal_t_table = np.linspace(0.05, 13.8, n_t)

    n_filters = 3
    ssp_obs_photflux_table = np.random.uniform(0, 1, size=(n_met, n_age, n_filters))

    n_filter_wave = 200
    _lam = np.linspace(1_000, 10_000, n_filter_wave)
    filter_waves = np.array([_lam for __ in range(n_filters)])
    filter_trans = np.ones((n_filters, n_filter_wave))

    n_gals = 250
    gal_sfr_table = np.random.uniform(0, 1, size=(n_gals, n_t))

    z_obs = 0.5

    ran_key = jran.PRNGKey(0)

    res = get_obs_photometry_singlez(
        ran_key,
        filter_waves,
        filter_trans,
        ssp_obs_photflux_table,
        ssp_lgmet,
        ssp_lg_age,
        gal_t_table,
        gal_sfr_table,
        DEFAULT_LGFBURST_U_PARAMS,
        DEFAULT_BURSTSHAPE_U_PARAMS,
        DEFAULT_LGAV_U_PARAMS,
        DEFAULT_DUST_DELTA_U_PARAMS,
        DEFAULT_BORIS_PARAMS,
        DEFAULT_COSMOLOGY,
        z_obs,
    )

    (
        weights,
        lgmet_weights,
        smooth_age_weights,
        bursty_age_weights,
        frac_trans,
        gal_obsflux_nodust,
        gal_obsflux,
    ) = res

    assert weights.shape == (n_gals, n_met, n_age)
    assert np.all(np.isfinite(weights))
    assert np.allclose(1.0, np.sum(weights, axis=(1, 2)), atol=1e-3)

    assert lgmet_weights.shape == (n_gals, n_met)
    assert np.all(np.isfinite(lgmet_weights))
    assert np.allclose(1.0, np.sum(lgmet_weights, axis=1), atol=1e-3)

    assert smooth_age_weights.shape == (n_gals, n_age)
    assert np.all(np.isfinite(smooth_age_weights))
    assert np.allclose(1.0, np.sum(smooth_age_weights, axis=1), atol=1e-3)

    assert bursty_age_weights.shape == (n_gals, n_age)
    assert np.all(np.isfinite(bursty_age_weights))
    assert np.allclose(1.0, np.sum(bursty_age_weights, axis=1), atol=1e-3)

    assert frac_trans.shape == (n_gals, n_age, n_filters)
    assert np.all(np.isfinite(frac_trans))
    assert np.all(frac_trans >= 0)
    assert np.all(frac_trans <= 1)

    assert gal_obsflux_nodust.shape == (n_gals, n_filters)
    assert np.all(np.isfinite(gal_obsflux_nodust))
    assert np.all(gal_obsflux_nodust >= 0)
    assert np.any(gal_obsflux_nodust > 0)

    assert gal_obsflux.shape == (n_gals, n_filters)
    assert np.all(np.isfinite(gal_obsflux))
    assert np.all(gal_obsflux >= 0)
    assert np.any(gal_obsflux > 0)

    assert np.all(gal_obsflux <= gal_obsflux_nodust)
    assert np.any(gal_obsflux < gal_obsflux_nodust)
