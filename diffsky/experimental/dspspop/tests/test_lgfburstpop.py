"""
"""
import numpy as np
from ..lgfburstpop import _get_lgfburst_galpop_from_params
from ..lgfburstpop import _get_lgfburst_galpop_from_u_params
from ..lgfburstpop import DEFAULT_LGFBURST_U_PARAMS, DEFAULT_LGFBURST_PARAMS
from ..lgfburstpop import _get_bounded_lgfburst_params, _get_unbounded_lgfburst_params


def test_get_bursty_age_weights_pop_evaluates():
    n_gals = 500
    gal_logsm = np.random.uniform(0, 10, size=(n_gals,))
    gal_logssfr = np.random.uniform(-12, -8, size=(n_gals,))

    gal_lgfburst = _get_lgfburst_galpop_from_params(
        gal_logsm, gal_logssfr, DEFAULT_LGFBURST_PARAMS
    )
    assert gal_lgfburst.shape == (n_gals,)


def test_get_bursty_age_weights_pop_u_param_inversion():
    assert np.allclose(
        DEFAULT_LGFBURST_PARAMS,
        _get_bounded_lgfburst_params(DEFAULT_LGFBURST_U_PARAMS),
        rtol=1e-3,
    )

    inferred_default_params = _get_bounded_lgfburst_params(
        _get_unbounded_lgfburst_params(DEFAULT_LGFBURST_PARAMS)
    )
    assert np.allclose(DEFAULT_LGFBURST_PARAMS, inferred_default_params, rtol=1e-3)

    n_gals = 500
    gal_logsm = np.random.uniform(0, 10, size=(n_gals,))
    gal_logssfr = np.random.uniform(-12, -8, size=(n_gals,))

    gal_lgfburst = _get_lgfburst_galpop_from_params(
        gal_logsm, gal_logssfr, DEFAULT_LGFBURST_PARAMS
    )
    assert gal_lgfburst.shape == (n_gals,)
    assert np.all(np.isfinite(gal_lgfburst))

    gal_lgfburst_u = _get_lgfburst_galpop_from_u_params(
        gal_logsm, gal_logssfr, DEFAULT_LGFBURST_U_PARAMS
    )
    assert gal_lgfburst_u.shape == (n_gals,)
    assert np.all(np.isfinite(gal_lgfburst_u))

    assert np.allclose(gal_lgfburst, gal_lgfburst_u, rtol=1e-4)
