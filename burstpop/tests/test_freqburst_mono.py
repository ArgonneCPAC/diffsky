""" """

import numpy as np
from jax import nn
from jax import random as jran

from ..freqburst_mono import (
    DEFAULT_FREQBURST_PARAMS,
    DEFAULT_FREQBURST_U_PARAMS,
    FREQBURST_PBOUNDS,
    SUFQB_BOUNDS,
    ZEROBURST_FREQBURST_PARAMS,
    FreqburstUParams,
    get_bounded_freqburst_params,
    get_freqburst_from_freqburst_params,
    get_freqburst_from_freqburst_u_params,
    get_unbounded_freqburst_params,
)

TOL = 1e-2
EPSILON = 1e-5


def test_default_params_are_in_bounds():

    gen = zip(DEFAULT_FREQBURST_PARAMS, DEFAULT_FREQBURST_PARAMS._fields)
    for val, key in gen:
        bound = getattr(FREQBURST_PBOUNDS, key)
        assert bound[0] < val < bound[1]


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


def test_get_freqburst_from_freqburst_params_fails_when_passing_u_params():
    logsm, logssfr = 10.0, -11.0

    try:
        get_freqburst_from_freqburst_params(DEFAULT_FREQBURST_U_PARAMS, logsm, logssfr)
        raise NameError(
            "get_freqburst_from_freqburst_params should not accept u_params"
        )
    except AttributeError:
        pass


def test_get_freqburst_from_freqburst_u_params_fails_when_passing_params():
    logsm, logssfr = 10.0, -11.0

    try:
        get_freqburst_from_freqburst_u_params(DEFAULT_FREQBURST_PARAMS, logsm, logssfr)
        raise NameError(
            "get_freqburst_from_freqburst_u_params should not accept params"
        )
    except AttributeError:
        pass


def test_get_freqburst_from_freqburst_params_evaluates():
    ran_key = jran.PRNGKey(0)
    n_gals = 500
    ran_key, logsm_key, logssfr_key = jran.split(ran_key, 3)
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=(n_gals,))
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=(n_gals,))

    gal_freqburst = get_freqburst_from_freqburst_params(
        DEFAULT_FREQBURST_PARAMS, logsm, logssfr
    )
    assert gal_freqburst.shape == (n_gals,)


def test_freqburst_u_param_inversion():
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

    gal_freqburst = get_freqburst_from_freqburst_params(
        DEFAULT_FREQBURST_PARAMS, logsm, logssfr
    )
    assert gal_freqburst.shape == (n_gals,)
    assert np.all(np.isfinite(gal_freqburst))

    gal_freqburst_u = get_freqburst_from_freqburst_u_params(
        DEFAULT_FREQBURST_U_PARAMS, logsm, logssfr
    )
    assert gal_freqburst_u.shape == (n_gals,)
    assert np.all(np.isfinite(gal_freqburst_u))

    assert np.allclose(gal_freqburst, gal_freqburst_u, rtol=1e-4)


def test_get_freqburst_from_freqburst_params_is_monotonic_with_logsm_and_logssfr():
    n_tests = 100
    ran_key = jran.PRNGKey(0)

    n_gals = 50
    logsmarr = np.linspace(0, 20, n_gals)
    logssfrarr = np.linspace(-20, 20, n_gals)
    n_pars = len(DEFAULT_FREQBURST_PARAMS)
    ZZ = np.zeros(n_gals)

    for __ in range(n_tests):
        ran_key, u_p_key = jran.split(ran_key, 2)

        ran_u_params = jran.uniform(u_p_key, minval=-10, maxval=10, shape=(n_pars,))
        freqb_params = get_bounded_freqburst_params(FreqburstUParams(*ran_u_params))

        sufq_max = SUFQB_BOUNDS[1]
        fqb_max = nn.softplus(sufq_max)
        assert fqb_max < 1

        for logssfr in logssfrarr:
            fqb = get_freqburst_from_freqburst_params(
                freqb_params, logsmarr, logssfr + ZZ
            )
            assert fqb.shape == (n_gals,)
            assert np.all(np.isfinite(fqb))
            assert np.all(fqb >= 0.0)
            assert np.all(fqb <= fqb_max)
            assert np.all(np.diff(fqb) <= EPSILON)

        for logsm in logsmarr:
            fqb = get_freqburst_from_freqburst_params(
                freqb_params, logsm + ZZ, logssfrarr
            )
            assert fqb.shape == (n_gals,)
            assert np.all(np.isfinite(fqb))
            assert np.all(fqb >= 0.0)
            assert np.all(fqb <= fqb_max)
            assert np.all(np.diff(fqb) >= -EPSILON)


def test_zeroburst_params_are_in_bounds():

    gen = zip(ZEROBURST_FREQBURST_PARAMS, ZEROBURST_FREQBURST_PARAMS._fields)
    for val, key in gen:
        bound = getattr(FREQBURST_PBOUNDS, key)
        assert bound[0] < val < bound[1]


def test_zeroburst_params_are_invertible():
    u_params = get_unbounded_freqburst_params(ZEROBURST_FREQBURST_PARAMS)
    params = get_bounded_freqburst_params(u_params)
    for p, p_orig in zip(ZEROBURST_FREQBURST_PARAMS, params):
        assert np.all(np.isfinite(p))
        assert np.allclose(p, p_orig, rtol=1e-4)


def test_zeroburst_params_produce_zero_burstiness():
    ran_key = jran.key(0)
    sm_key, ssfr_key = jran.split(ran_key, 2)
    n_gals = 2_000
    logsmarr = jran.uniform(sm_key, minval=5, maxval=13, shape=(n_gals,))
    logssfr = jran.uniform(ssfr_key, minval=-14, maxval=-5, shape=(n_gals,))

    fqb = get_freqburst_from_freqburst_params(
        ZEROBURST_FREQBURST_PARAMS, logsmarr, logssfr
    )
    assert fqb.shape == (n_gals,)
    assert np.all(np.isfinite(fqb))
    assert np.all(fqb >= 0.0)
    assert np.all(fqb < 0.1), fqb.max()

    sufq_max = SUFQB_BOUNDS[1]
    fqb_max = nn.softplus(sufq_max)
    assert np.all(fqb <= fqb_max)
