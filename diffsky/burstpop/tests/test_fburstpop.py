"""
"""

import numpy as np
from jax import random as jran

from ..fburstpop import (
    DEFAULT_FBURSTPOP_PARAMS,
    DEFAULT_FBURSTPOP_U_PARAMS,
    FBURSTPOP_PBOUNDS,
    ZEROBURST_FBURSTPOP_PARAMS,
    get_bounded_fburstpop_params,
    get_lgfburst_from_fburstpop_params,
    get_lgfburst_from_fburstpop_u_params,
    get_unbounded_fburstpop_params,
)

TOL = 1e-2


def test_default_params_are_in_bounds():

    gen = zip(DEFAULT_FBURSTPOP_PARAMS, DEFAULT_FBURSTPOP_PARAMS._fields)
    for val, key in gen:
        bound = getattr(FBURSTPOP_PBOUNDS, key)
        assert bound[0] < val < bound[1]


def test_param_u_param_names_propagate_properly():
    gen = zip(DEFAULT_FBURSTPOP_U_PARAMS._fields, DEFAULT_FBURSTPOP_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = get_bounded_fburstpop_params(DEFAULT_FBURSTPOP_U_PARAMS)
    assert set(inferred_default_params._fields) == set(DEFAULT_FBURSTPOP_PARAMS._fields)

    inferred_default_u_params = get_unbounded_fburstpop_params(DEFAULT_FBURSTPOP_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        DEFAULT_FBURSTPOP_U_PARAMS._fields
    )


def test_get_bounded_fburstpop_params_fails_when_passing_params():
    try:
        get_bounded_fburstpop_params(DEFAULT_FBURSTPOP_PARAMS)
        raise NameError("get_bounded_fburstpop_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_fburstpop_params_fails_when_passing_u_params():
    try:
        get_unbounded_fburstpop_params(DEFAULT_FBURSTPOP_U_PARAMS)
        raise NameError("get_unbounded_fburstpop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_lgfburst_from_fburstpop_params_fails_when_passing_u_params():
    logsm, logssfr = 10.0, -11.0

    try:
        get_lgfburst_from_fburstpop_params(DEFAULT_FBURSTPOP_U_PARAMS, logsm, logssfr)
        raise NameError("get_lgfburst_from_fburstpop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_lgfburst_from_fburstpop_u_params_fails_when_passing_params():
    logsm, logssfr = 10.0, -11.0

    try:
        get_lgfburst_from_fburstpop_u_params(DEFAULT_FBURSTPOP_PARAMS, logsm, logssfr)
        raise NameError("get_lgfburst_from_fburstpop_u_params should not accept params")
    except AttributeError:
        pass


def test_get_bursty_age_weights_pop_evaluates():
    ran_key = jran.PRNGKey(0)
    n_gals = 500
    ran_key, logsm_key, logssfr_key = jran.split(ran_key, 3)
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=(n_gals,))
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=(n_gals,))

    gal_lgfburst = get_lgfburst_from_fburstpop_params(
        DEFAULT_FBURSTPOP_PARAMS, logsm, logssfr
    )
    assert gal_lgfburst.shape == (n_gals,)


def test_get_bursty_age_weights_pop_u_param_inversion():
    assert np.allclose(
        DEFAULT_FBURSTPOP_PARAMS,
        get_bounded_fburstpop_params(DEFAULT_FBURSTPOP_U_PARAMS),
        rtol=TOL,
    )

    inferred_default_params = get_bounded_fburstpop_params(
        get_unbounded_fburstpop_params(DEFAULT_FBURSTPOP_PARAMS)
    )
    assert np.allclose(DEFAULT_FBURSTPOP_PARAMS, inferred_default_params, rtol=TOL)

    ran_key = jran.PRNGKey(0)
    n_gals = 500
    ran_key, logsm_key, logssfr_key = jran.split(ran_key, 3)
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=(n_gals,))
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=(n_gals,))

    gal_lgfburst = get_lgfburst_from_fburstpop_params(
        DEFAULT_FBURSTPOP_PARAMS, logsm, logssfr
    )
    assert gal_lgfburst.shape == (n_gals,)
    assert np.all(np.isfinite(gal_lgfburst))

    gal_lgfburst_u = get_lgfburst_from_fburstpop_u_params(
        DEFAULT_FBURSTPOP_U_PARAMS, logsm, logssfr
    )
    assert gal_lgfburst_u.shape == (n_gals,)
    assert np.all(np.isfinite(gal_lgfburst_u))

    assert np.allclose(gal_lgfburst, gal_lgfburst_u, rtol=1e-4)


def test_zeroburst_params_are_in_bounds():

    gen = zip(ZEROBURST_FBURSTPOP_PARAMS, ZEROBURST_FBURSTPOP_PARAMS._fields)
    for val, key in gen:
        bound = getattr(FBURSTPOP_PBOUNDS, key)
        assert bound[0] < val < bound[1]


def test_zeroburst_params_produce_zero_burstiness():
    ran_key = jran.key(0)
    sm_key, ssfr_key = jran.split(ran_key, 2)
    n_gals = 2_000
    logsmarr = jran.uniform(sm_key, minval=5, maxval=13, shape=(n_gals,))
    logssfr = jran.uniform(ssfr_key, minval=-14, maxval=-5, shape=(n_gals,))

    fb = 10 ** get_lgfburst_from_fburstpop_params(
        ZEROBURST_FBURSTPOP_PARAMS, logsmarr, logssfr
    )
    assert fb.shape == (n_gals,)
    assert np.all(np.isfinite(fb))
    assert np.all(fb > 0.0)
    assert np.all(fb < 1e-4)
