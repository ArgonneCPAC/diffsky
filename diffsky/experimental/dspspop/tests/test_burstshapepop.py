"""
"""
import numpy as np
from dsps.sfh.diffburst import _pureburst_age_weights_from_u_params
from jax import jit as jjit
from jax import vmap

from ..burstshapepop import (
    DEFAULT_BURSTSHAPE_PARAMS,
    DEFAULT_BURSTSHAPE_U_PARAMS,
    _get_bounded_burstshape_params,
    _get_burstshape_galpop_from_params,
    _get_burstshape_galpop_from_u_params,
    _get_unbounded_burstshape_params,
)

_B = (None, 0, 0)
_pureburst_age_weights_from_u_params_vmap = jjit(
    vmap(_pureburst_age_weights_from_u_params, in_axes=_B)
)


def test_get_bursty_age_weights_pop_evaluates():
    n_gals = 500
    gal_logsm = np.random.uniform(0, 10, size=(n_gals,))
    gal_logssfr = np.random.uniform(-12, -8, size=(n_gals,))

    n_age = 107
    lgyr = np.linspace(5.5, 10.25, n_age)

    u_lgyr_peak, u_lgyr_max = _get_burstshape_galpop_from_params(
        gal_logsm, gal_logssfr, DEFAULT_BURSTSHAPE_PARAMS
    )
    assert u_lgyr_peak.shape == (n_gals,)
    assert u_lgyr_max.shape == (n_gals,)

    age_weights = _pureburst_age_weights_from_u_params_vmap(
        lgyr, u_lgyr_peak, u_lgyr_max
    )
    assert age_weights.shape == (n_gals, n_age)
    assert np.all(np.isfinite(age_weights))
    assert np.allclose(np.sum(age_weights, axis=1), 1.0, rtol=1e-4)
    assert np.all(age_weights <= 1)
    assert np.all(age_weights >= 0)


def test_get_bursty_age_weights_pop_u_param_inversion():
    assert np.allclose(
        DEFAULT_BURSTSHAPE_PARAMS,
        _get_bounded_burstshape_params(DEFAULT_BURSTSHAPE_U_PARAMS),
        atol=0.05,
    )

    inferred_default_params = _get_bounded_burstshape_params(
        _get_unbounded_burstshape_params(DEFAULT_BURSTSHAPE_PARAMS)
    )
    assert np.allclose(DEFAULT_BURSTSHAPE_PARAMS, inferred_default_params, rtol=1e-3)

    n_gals = 10_000
    gal_logsm = np.random.uniform(0, 10, size=(n_gals,))
    gal_logssfr = np.random.uniform(-12, -8, size=(n_gals,))

    u_lgyr_peak, u_lgyr_max = _get_burstshape_galpop_from_params(
        gal_logsm, gal_logssfr, DEFAULT_BURSTSHAPE_PARAMS
    )
    assert u_lgyr_peak.shape == (n_gals,)
    assert u_lgyr_max.shape == (n_gals,)
    assert np.all(np.isfinite(u_lgyr_peak))
    assert np.all(np.isfinite(u_lgyr_max))

    u_lgyr_peak_u, u_lgyr_max_u = _get_burstshape_galpop_from_u_params(
        gal_logsm, gal_logssfr, DEFAULT_BURSTSHAPE_U_PARAMS
    )
    assert u_lgyr_peak_u.shape == (n_gals,)
    assert u_lgyr_max_u.shape == (n_gals,)
    assert np.all(np.isfinite(u_lgyr_peak_u))
    assert np.all(np.isfinite(u_lgyr_max_u))

    assert np.allclose(u_lgyr_max, u_lgyr_max_u, atol=0.05)
    assert np.allclose(u_lgyr_peak, u_lgyr_peak_u, atol=0.05)
