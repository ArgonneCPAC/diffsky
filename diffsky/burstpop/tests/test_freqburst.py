"""
"""

import numpy as np
from jax import random as jran

from ..freqburst import (
    DEFAULT_FREQBURST_PARAMS,
    DEFAULT_FREQBURST_U_PARAMS,
    get_bounded_freqburst_params,
    get_lgfreqburst_from_freqburst_params,
    get_lgfreqburst_from_freqburst_u_params,
    get_unbounded_freqburst_params,
)

TOL = 1e-2


def test_param_u_param_names_propagate_properly():
    gen = zip(DEFAULT_FREQBURST_U_PARAMS._fields, DEFAULT_FREQBURST_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = get_bounded_freqburst_params(DEFAULT_FREQBURST_U_PARAMS)
    assert set(inferred_default_params._fields) == set(DEFAULT_FREQBURST_PARAMS._fields)

    inferred_default_u_params = get_unbounded_freqburst_params(DEFAULT_FREQBURST_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        DEFAULT_FREQBURST_U_PARAMS._fields
    )


def test_get_bounded_freqburst_params_fails_when_passing_params():
    try:
        get_bounded_freqburst_params(DEFAULT_FREQBURST_PARAMS)
        raise NameError("get_bounded_freqburst_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_freqburst_params_fails_when_passing_u_params():
    try:
        get_unbounded_freqburst_params(DEFAULT_FREQBURST_U_PARAMS)
        raise NameError("get_unbounded_freqburst_params should not accept u_params")
    except AttributeError:
        pass


def test_get_lgfreqburst_from_freqburst_params_fails_when_passing_u_params():
    logsm, logssfr = 10.0, -11.0

    try:
        get_lgfreqburst_from_freqburst_params(
            DEFAULT_FREQBURST_U_PARAMS, logsm, logssfr
        )
        raise NameError(
            "get_lgfreqburst_from_freqburst_params should not accept u_params"
        )
    except AttributeError:
        pass


def test_get_lgfreqburst_from_freqburst_u_params_fails_when_passing_params():
    logsm, logssfr = 10.0, -11.0

    try:
        get_lgfreqburst_from_freqburst_u_params(
            DEFAULT_FREQBURST_PARAMS, logsm, logssfr
        )
        raise NameError(
            "get_lgfreqburst_from_freqburst_u_params should not accept params"
        )
    except AttributeError:
        pass


def test_get_bursty_age_weights_evaluates():
    ran_key = jran.PRNGKey(0)
    n_gals = 500
    ran_key, logsm_key, logssfr_key = jran.split(ran_key, 3)
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=(n_gals,))
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=(n_gals,))

    gal_lgfreqburst = get_lgfreqburst_from_freqburst_params(
        DEFAULT_FREQBURST_PARAMS, logsm, logssfr
    )
    assert gal_lgfreqburst.shape == (n_gals,)


def test_get_bursty_age_weights_u_param_inversion():
    assert np.allclose(
        DEFAULT_FREQBURST_PARAMS,
        get_bounded_freqburst_params(DEFAULT_FREQBURST_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = get_bounded_freqburst_params(
        get_unbounded_freqburst_params(DEFAULT_FREQBURST_PARAMS)
    )
    assert np.allclose(DEFAULT_FREQBURST_PARAMS, inferred_default_params, rtol=TOL)

    ran_key = jran.PRNGKey(0)
    n_gals = 500
    ran_key, logsm_key, logssfr_key = jran.split(ran_key, 3)
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=(n_gals,))
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=(n_gals,))

    gal_lgfreqburst = get_lgfreqburst_from_freqburst_params(
        DEFAULT_FREQBURST_PARAMS, logsm, logssfr
    )
    assert gal_lgfreqburst.shape == (n_gals,)
    assert np.all(np.isfinite(gal_lgfreqburst))

    gal_lgfreqburst_u = get_lgfreqburst_from_freqburst_u_params(
        DEFAULT_FREQBURST_U_PARAMS, logsm, logssfr
    )
    assert gal_lgfreqburst_u.shape == (n_gals,)
    assert np.all(np.isfinite(gal_lgfreqburst_u))

    assert np.allclose(gal_lgfreqburst, gal_lgfreqburst_u, rtol=1e-4)
