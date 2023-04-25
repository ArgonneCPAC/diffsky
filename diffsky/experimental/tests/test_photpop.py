"""
"""
from jax import random as jran
import numpy as np
from ..photpop import get_obs_photometry_singlez
from dsps.cosmology import DEFAULT_COSMOLOGY
from ..nagaraj22_dust import TAU_PARAMS, DELTA_PARAMS


def test_photpop_evaluates():
    n_met, n_age = 20, 50
    ssp_lgmet = np.linspace(-4, 0, n_met)
    ssp_lg_age = np.linspace(5.5, 10.5, n_age) - 9

    n_t = 100
    gal_t_table = np.linspace(0.05, 13.8, n_t)

    n_filters = 3
    ssp_obsmag_table = np.random.uniform(0, 1, size=(n_met, n_age, n_filters))

    n_filter_wave = 200
    _lam = np.linspace(1_000, 10_000, n_filter_wave)
    filter_waves = np.array([_lam for __ in range(n_filters)])
    filter_trans = np.ones((n_filters, n_filter_wave))

    n_gals = 50
    gal_sfr_table = np.random.uniform(0, 1, size=(n_gals, n_t))

    z_obs = 0.5

    burst_params = (0.0,)
    att_curve_params = (TAU_PARAMS, DELTA_PARAMS)
    ran_key = jran.PRNGKey(0)

    res = get_obs_photometry_singlez(
        ran_key,
        filter_waves,
        filter_trans,
        ssp_obsmag_table,
        ssp_lgmet,
        ssp_lg_age,
        gal_t_table,
        gal_sfr_table,
        burst_params,
        att_curve_params,
        DEFAULT_COSMOLOGY,
        z_obs,
    )
    weights, lgmet_weights, smooth_age_weights, bursty_age_weights = res
    assert weights.shape == (n_gals, n_met, n_age)
    assert np.all(np.isfinite(weights))

    assert bursty_age_weights.shape == (n_gals, n_age)
    assert np.all(np.isfinite(bursty_age_weights))
    assert np.allclose(np.sum(bursty_age_weights, axis=1), 1.0, rtol=1e-4)
