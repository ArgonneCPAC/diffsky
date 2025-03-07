"""
"""

import numpy as np
from jax import random as jran

from .. import ssp_errors as sspe


def test_param_u_param_names_propagate_properly():
    """Each unbounded param should have `u_` in front of corresponding param"""
    gen = zip(sspe.DEFAULT_SSPERR_U_PARAMS._fields, sspe.DEFAULT_SSPERR_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key


def test_default_u_params_and_params_are_consistent():
    """Default unbounded parameters should agree with unbounding the default params"""
    gen = zip(sspe.DEFAULT_SSPERR_U_PARAMS._fields, sspe.DEFAULT_SSPERR_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key
    inferred_default_params = sspe.get_bounded_params(sspe.DEFAULT_SSPERR_U_PARAMS)
    assert set(inferred_default_params._fields) == set(
        sspe.DEFAULT_SSPERR_PARAMS._fields
    )

    inferred_default_u_params = sspe.get_unbounded_params(sspe.DEFAULT_SSPERR_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        sspe.DEFAULT_SSPERR_U_PARAMS._fields
    )


def test_default_params_are_in_bounds():
    """Default parameters should lie strictly within the bounds"""
    gen = zip(sspe.DEFAULT_SSPERR_PARAMS, sspe.SSPERR_PBOUNDS)
    for default, bounds in gen:
        assert bounds[0] < default < bounds[1]


def test_get_bounded_params_fails_when_passing_params():
    """Bounding function should fail when passing bounded parameters"""
    try:
        sspe.get_bounded_params(sspe.DEFAULT_SSPERR_PARAMS)
        raise NameError("get_bounded_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_u_params():
    """Unbounding function should fail when passing unbounded parameters"""
    try:
        sspe.get_unbounded_params(sspe.DEFAULT_SSPERR_U_PARAMS)
        raise NameError("get_unbounded_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    """Bounding and unbounding functions should be inverses of each other"""
    ran_key = jran.key(0)
    n_params = len(sspe.DEFAULT_SSPERR_PARAMS)

    n_tests = 100
    for __ in range(n_tests):
        test_key, ran_key = jran.split(ran_key, 2)
        uran = jran.uniform(test_key, minval=-100, maxval=100, shape=(n_params,))
        u_params = sspe.DEFAULT_SSPERR_U_PARAMS._make(uran)
        params = sspe.get_bounded_params(u_params)
        u_params2 = sspe.get_unbounded_params(params)
        assert np.allclose(u_params, u_params2, rtol=1e-3)


def test_ssp_flux_factor_fails_when_passing_u_params():
    """The ssp_flux_factor function should fail when passing unbounded parameters"""
    wave = np.logspace(3, 4, 100)
    try:
        sspe.ssp_flux_factor(sspe.DEFAULT_SSPERR_U_PARAMS, wave)
        raise NameError("ssp_flux_factor should not accept u_params")
    except AttributeError:
        pass


def test_flux_factor_kern_default_is_well_behaved():
    """The ssp_flux_factor function should return sensible values for default params"""
    wave = np.logspace(3, 4, 100)
    ff = sspe.ssp_flux_factor(sspe.DEFAULT_SSPERR_PARAMS, wave)
    assert np.all(ff >= 10 ** sspe.SSPERR_PBOUNDS.ssp_ff_ylo[0])
    assert np.all(ff <= 10 ** sspe.SSPERR_PBOUNDS.ssp_ff_ylo[1])


def test_flux_factor_kern_rando_is_well_behaved():
    """The ssp_flux_factor function should return sensible values for random params"""
    ran_key = jran.key(0)
    wave = np.logspace(0, 10, 100)
    n_params = len(sspe.DEFAULT_SSPERR_PARAMS)

    n_tests = 100
    for __ in range(n_tests):
        test_key, ran_key = jran.split(ran_key, 2)
        uran = jran.uniform(test_key, minval=-100, maxval=100, shape=(n_params,))
        u_params = sspe.DEFAULT_SSPERR_U_PARAMS._make(uran)
        params = sspe.get_bounded_params(u_params)
        ff = sspe.ssp_flux_factor(params, wave)
        assert np.all(ff >= 10 ** sspe.SSPERR_PBOUNDS.ssp_ff_ylo[0])
        assert np.all(ff <= 10 ** sspe.SSPERR_PBOUNDS.ssp_ff_ylo[1])
        assert not np.allclose(ff, 1.0)
