"""
"""
import numpy as np
from ..lgavpop import _get_lgav_galpop_from_params
from ..lgavpop import _get_lgav_galpop_from_u_params
from ..lgavpop import DEFAULT_LGAV_U_PARAMS, DEFAULT_LGAV_PARAMS
from ..lgavpop import _get_bounded_lgav_params, _get_unbounded_lgav_params


def test_get_bursty_age_weights_pop_evaluates():
    n_gals = 500
    gal_logsm = np.random.uniform(0, 10, size=(n_gals,))
    gal_logssfr = np.random.uniform(-12, -8, size=(n_gals,))

    gal_lgav = _get_lgav_galpop_from_params(gal_logsm, gal_logssfr, DEFAULT_LGAV_PARAMS)
    assert gal_lgav.shape == (n_gals,)


def test_get_bursty_age_weights_pop_u_param_inversion():
    assert np.allclose(
        DEFAULT_LGAV_PARAMS,
        _get_bounded_lgav_params(DEFAULT_LGAV_U_PARAMS),
        rtol=1e-3,
    )

    inferred_default_params = _get_bounded_lgav_params(
        _get_unbounded_lgav_params(DEFAULT_LGAV_PARAMS)
    )
    assert np.allclose(DEFAULT_LGAV_PARAMS, inferred_default_params, rtol=1e-3)

    n_gals = 500
    gal_logsm = np.random.uniform(0, 10, size=(n_gals,))
    gal_logssfr = np.random.uniform(-12, -8, size=(n_gals,))

    gal_lgav = _get_lgav_galpop_from_params(gal_logsm, gal_logssfr, DEFAULT_LGAV_PARAMS)
    assert gal_lgav.shape == (n_gals,)
    assert np.all(np.isfinite(gal_lgav))

    gal_lgav_u = _get_lgav_galpop_from_u_params(
        gal_logsm, gal_logssfr, DEFAULT_LGAV_U_PARAMS
    )
    assert gal_lgav_u.shape == (n_gals,)
    assert np.all(np.isfinite(gal_lgav_u))

    assert np.allclose(gal_lgav, gal_lgav_u, rtol=1e-4)
