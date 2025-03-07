"""
"""

import numpy as np
from jax import random as jran

from ...systematics import ssp_errors
from .. import ssp_err_pop


def test_param_u_param_names_propagate_properly():
    """Each unbounded param should have `u_` in front of corresponding param"""
    gen = zip(
        ssp_err_pop.DEFAULT_SSP_ERR_POP_U_PARAMS._fields,
        ssp_err_pop.DEFAULT_SSP_ERR_POP_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key


def test_default_u_params_and_params_are_consistent():
    """Default unbounded parameters should agree with unbounding the default params"""
    gen = zip(
        ssp_err_pop.DEFAULT_SSP_ERR_POP_U_PARAMS._fields,
        ssp_err_pop.DEFAULT_SSP_ERR_POP_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key
    inferred_default_params = ssp_err_pop.get_bounded_params(
        ssp_err_pop.DEFAULT_SSP_ERR_POP_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        ssp_err_pop.DEFAULT_SSP_ERR_POP_PARAMS._fields
    )

    inferred_default_u_params = ssp_err_pop.get_unbounded_params(
        ssp_err_pop.DEFAULT_SSP_ERR_POP_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        ssp_err_pop.DEFAULT_SSP_ERR_POP_U_PARAMS._fields
    )


def test_default_params_are_in_bounds():
    """Default parameters should lie strictly within the bounds"""
    gen = zip(ssp_err_pop.DEFAULT_SSP_ERR_POP_PARAMS, ssp_err_pop.SSPERR_POP_PBOUNDS)
    for default, bounds in gen:
        assert bounds[0] < default < bounds[1]


def test_get_bounded_params_fails_when_passing_params():
    """Bounding function should fail when passing bounded parameters"""
    try:
        ssp_err_pop.get_bounded_params(ssp_err_pop.DEFAULT_SSP_ERR_POP_PARAMS)
        raise NameError("get_bounded_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_u_params():
    """Unbounding function should fail when passing unbounded parameters"""
    try:
        ssp_err_pop.get_unbounded_params(ssp_err_pop.DEFAULT_SSP_ERR_POP_U_PARAMS)
        raise NameError("get_unbounded_params should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    """Bounding and unbounding functions should be inverses of each other"""
    ran_key = jran.key(0)
    n_params = len(ssp_err_pop.DEFAULT_SSP_ERR_POP_PARAMS)

    n_tests = 100
    for __ in range(n_tests):
        test_key, ran_key = jran.split(ran_key, 2)
        uran = jran.uniform(test_key, minval=-100, maxval=100, shape=(n_params,))
        u_params = ssp_err_pop.DEFAULT_SSP_ERR_POP_U_PARAMS._make(uran)
        params = ssp_err_pop.get_bounded_params(u_params)
        u_params2 = ssp_err_pop.get_unbounded_params(params)
        assert np.allclose(u_params, u_params2, rtol=1e-3)


def test_get_flux_factor_from_lgssfr_kern():
    ssp_err_pop_params = ssp_err_pop.DEFAULT_SSP_ERR_POP_PARAMS
    n_gals = 300
    wave = 4_000.0
    lgssfr = np.linspace(-12, -8, n_gals)
    flux_factor = ssp_err_pop.get_flux_factor_from_lgssfr_kern(
        ssp_err_pop_params, lgssfr, wave
    )
    assert np.all(np.isfinite(flux_factor))
    assert np.all(flux_factor >= 1.0 + ssp_errors.SSPERR_FF_BOUNDS[0])
    assert np.all(flux_factor <= 1.0 + ssp_errors.SSPERR_FF_BOUNDS[1])
