"""
"""
import numpy as np
from ..dust_deltapop import _get_dust_delta_galpop_from_params
from ..dust_deltapop import _get_dust_delta_galpop_from_u_params
from ..dust_deltapop import DEFAULT_DUST_DELTA_U_PARAMS, DEFAULT_DUST_DELTA_PARAMS
from ..dust_deltapop import (
    _get_bounded_dust_delta_params,
    _get_unbounded_dust_delta_params,
)


def test_get_bursty_age_weights_pop_evaluates():
    n_gals = 500
    gal_logsm = np.random.uniform(0, 10, size=(n_gals,))
    gal_logssfr = np.random.uniform(-12, -8, size=(n_gals,))

    gal_dust_delta = _get_dust_delta_galpop_from_params(
        gal_logsm, gal_logssfr, DEFAULT_DUST_DELTA_PARAMS
    )
    assert gal_dust_delta.shape == (n_gals,)


def test_get_bursty_age_weights_pop_u_param_inversion():
    assert np.allclose(
        DEFAULT_DUST_DELTA_PARAMS,
        _get_bounded_dust_delta_params(DEFAULT_DUST_DELTA_U_PARAMS),
        rtol=1e-3,
    )

    inferred_default_params = _get_bounded_dust_delta_params(
        _get_unbounded_dust_delta_params(DEFAULT_DUST_DELTA_PARAMS)
    )
    assert np.allclose(DEFAULT_DUST_DELTA_PARAMS, inferred_default_params, rtol=1e-3)

    n_gals = 500
    gal_logsm = np.random.uniform(0, 10, size=(n_gals,))
    gal_logssfr = np.random.uniform(-12, -8, size=(n_gals,))

    gal_dust_delta = _get_dust_delta_galpop_from_params(
        gal_logsm, gal_logssfr, DEFAULT_DUST_DELTA_PARAMS
    )
    assert gal_dust_delta.shape == (n_gals,)
    assert np.all(np.isfinite(gal_dust_delta))

    gal_dust_delta_u = _get_dust_delta_galpop_from_u_params(
        gal_logsm, gal_logssfr, DEFAULT_DUST_DELTA_U_PARAMS
    )
    assert gal_dust_delta_u.shape == (n_gals,)
    assert np.all(np.isfinite(gal_dust_delta_u))

    assert np.allclose(gal_dust_delta, gal_dust_delta_u, rtol=1e-4)
