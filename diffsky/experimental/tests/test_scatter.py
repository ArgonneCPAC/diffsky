"""
"""

from ..scatter import (
    DEFAULT_SCATTER_PARAMS,
    DEFAULT_SCATTER_U_PARAMS,
    SCATTER_PBOUNDS,
    get_unbounded_scatter_params,
    get_bounded_scatter_params
)

import numpy as np

from jax import random as jran


TOL = 1e-4


def test_dustpop_noise_u_param_inversion_default_params():
    u_params = get_unbounded_scatter_params(
        DEFAULT_SCATTER_PARAMS
    )
    assert np.all(np.isfinite(u_params))
    params = get_bounded_scatter_params(u_params)
    assert np.all(np.isfinite(params))
    assert np.allclose(params, DEFAULT_SCATTER_PARAMS)


def test_param_u_param_names_propagate_properly():
    gen = zip(
        DEFAULT_SCATTER_U_PARAMS._fields,
        DEFAULT_SCATTER_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = get_bounded_scatter_params(
        DEFAULT_SCATTER_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        DEFAULT_SCATTER_PARAMS._fields
    )

    inferred_default_u_params = get_unbounded_scatter_params(
        DEFAULT_SCATTER_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        DEFAULT_SCATTER_U_PARAMS._fields
    )


def test_get_bounded_params_fails_when_passing_params():
    try:
        get_bounded_scatter_params(DEFAULT_SCATTER_PARAMS)
        raise NameError("get_bounded_dustpop_scatter_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_params():
    try:
        get_unbounded_scatter_params(DEFAULT_SCATTER_U_PARAMS)
        raise NameError(
            "get_unbounded_dustpop_scatter_params should not accept u_params"
        )
    except AttributeError:
        pass


def test_default_params_are_in_bounds():

    gen = zip(
        DEFAULT_SCATTER_PARAMS, DEFAULT_SCATTER_PARAMS._fields
    )
    for val, key in gen:
        bound = getattr(SCATTER_PBOUNDS, key)
        assert bound[0] < val < bound[1]


def test_random_params_are_always_invertible():
    ran_key = jran.key(0)
    n_params = len(DEFAULT_SCATTER_PARAMS)
    n_tests = 100
    for __ in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        u_params = jran.uniform(test_key, minval=-100, maxval=100, shape=(n_params,))
        u_params = DEFAULT_SCATTER_U_PARAMS._make(u_params)
        params = get_bounded_scatter_params(u_params)
        assert np.all(np.isfinite(params))
        u_params2 = get_unbounded_scatter_params(params)
        assert np.all(np.isfinite(u_params2))
        assert np.allclose(u_params, u_params2, rtol=TOL)
