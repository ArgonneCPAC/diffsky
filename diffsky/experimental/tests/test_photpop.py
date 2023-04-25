"""
"""
import numpy as np
from ..photpop import get_obs_photometry_singlez
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.experimental.diffburst import DEFAULT_DBURST


def test_photpop_evaluates():
    n_met, n_age = 20, 50
    ssp_lgmet = np.linspace(-4, 0, n_met)
    ssp_lg_age = np.linspace(5.5, 10.5, n_age) - 9

    n_t = 100
    gal_t_table = np.linspace(0.05, 13.8, n_t)

    n_filters = 3
    ssp_obsmag_table = np.random.uniform(0, 1, size=(n_met, n_age, n_filters))

    n_gals = 50
    gal_sfr_table = np.random.uniform(0, 1, size=(n_gals, n_t))

    gal_fburst = np.random.uniform(0, 0.1, n_gals)
    gal_dburst = np.zeros(n_gals) + DEFAULT_DBURST

    z_obs = 0.5

    res = get_obs_photometry_singlez(
        ssp_obsmag_table,
        ssp_lgmet,
        ssp_lg_age,
        gal_t_table,
        gal_sfr_table,
        gal_fburst,
        gal_dburst,
        DEFAULT_COSMOLOGY,
        z_obs,
    )
    weights, lgmet_weights, smooth_age_weights, bursty_age_weights = res
    assert weights.shape == (n_gals, n_met, n_age)
    assert np.all(np.isfinite(weights))

    assert bursty_age_weights.shape == (n_gals, n_age)
    assert np.all(np.isfinite(bursty_age_weights))
    assert np.allclose(np.sum(bursty_age_weights, axis=1), 1.0, rtol=1e-4)
