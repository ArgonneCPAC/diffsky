"""
"""

import numpy as np
from jax import random as jran

from .. import threeroll_smhm as tsm


def test_param_u_param_names_propagate_properly():
    """Each unbounded param should have `u_` in front of corresponding param"""
    gen = zip(tsm.DEFAULT_U_PARAMS._fields, tsm.DEFAULT_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key


def test_default_u_params_and_params_are_consistent():
    """Default unbounded parameters should agree with unbounding the default params"""
    gen = zip(tsm.DEFAULT_U_PARAMS._fields, tsm.DEFAULT_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key
    inferred_default_params = tsm.get_bounded_params(tsm.DEFAULT_U_PARAMS)
    assert set(inferred_default_params._fields) == set(tsm.DEFAULT_PARAMS._fields)

    inferred_default_u_params = tsm.get_unbounded_params(tsm.DEFAULT_PARAMS)
    assert set(inferred_default_u_params._fields) == set(tsm.DEFAULT_U_PARAMS._fields)


def test_default_params_are_in_bounds():
    """Default parameters should lie strictly within the bounds"""
    gen = zip(tsm.DEFAULT_PARAMS, tsm.PBOUNDS)
    for default, bounds in gen:
        assert bounds[0] < default < bounds[1]


def test_get_bounded_params_fails_when_passing_params():
    """Bounding function should fail when passing bounded parameters"""
    try:
        tsm.get_bounded_params(tsm.DEFAULT_PARAMS)
        raise NameError("get_bounded_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_u_params():
    """Unbounding function should fail when passing unbounded parameters"""
    try:
        tsm.get_unbounded_params(tsm.DEFAULT_U_PARAMS)
        raise NameError("get_unbounded_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    """Bounding and unbounding functions should be inverses of each other"""
    ran_key = jran.key(0)
    n_params = len(tsm.DEFAULT_PARAMS)

    n_tests = 100
    for __ in range(n_tests):
        test_key, ran_key = jran.split(ran_key, 2)
        uran = jran.uniform(test_key, minval=-100, maxval=100, shape=(n_params,))
        u_params = tsm.DEFAULT_U_PARAMS._make(uran)
        params = tsm.get_bounded_params(u_params)
        u_params2 = tsm.get_unbounded_params(params)
        assert np.allclose(u_params, u_params2, rtol=1e-3)


def test_smhm_fails_when_passing_u_params():
    """The smhm_kernel function should fail when passing unbounded parameters"""
    logmh = np.linspace(5, 15, 200)
    try:
        tsm.smhm_kernel(tsm.DEFAULT_U_PARAMS, logmh)
        raise NameError("smhm_kernel should not accept u_params")
    except AttributeError:
        pass


def test_unbounded_smhm_fails_when_passing_params():
    """The smhm_kernel_u_params function should fail when passing bounded parameters"""
    logmh = np.linspace(5, 15, 200)
    try:
        tsm.smhm_kernel_u_params(tsm.DEFAULT_PARAMS, logmh)
        raise NameError("smhm_kernel_u_params should not accept params")
    except AttributeError:
        pass


def test_smhm_returns_finite_default_params():
    logmh = np.linspace(9, 15, 200)
    logsm = tsm.smhm_kernel(tsm.DEFAULT_PARAMS, logmh)
    assert np.all(np.isfinite(logsm))
    assert np.all(logsm < 16)
    assert np.all(logsm > 3)


def test_smhm_returns_finite_rando_params():
    """smhm_kernel_u_params should return sensible values for random params"""
    ran_key = jran.key(0)
    wave = np.logspace(0, 10, 100)
    n_params = len(tsm.DEFAULT_PARAMS)

    n_tests = 100
    for __ in range(n_tests):
        test_key, ran_key = jran.split(ran_key, 2)
        uran = jran.uniform(test_key, minval=-100, maxval=100, shape=(n_params,))
        u_params = tsm.DEFAULT_U_PARAMS._make(uran)
        params = tsm.get_bounded_params(u_params)
        logsm = tsm.smhm_kernel(params, wave)
        assert np.all(np.isfinite(logsm))
