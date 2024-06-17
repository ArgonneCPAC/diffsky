"""
"""
import numpy as np
from dsps.sfh.diffburst import (
    _pureburst_age_weights_from_params,
    _pureburst_age_weights_from_u_params,
)
from jax import random as jran

from .. import tburstpop as tbp

TOL = 1e-2


def test_param_u_param_names_propagate_properly():
    gen = zip(
        tbp.DEFAULT_TBURSTPOP_U_PARAMS._fields, tbp.DEFAULT_TBURSTPOP_PARAMS._fields
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = tbp.get_bounded_tburstpop_params(
        tbp.DEFAULT_TBURSTPOP_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        tbp.DEFAULT_TBURSTPOP_PARAMS._fields
    )

    inferred_default_u_params = tbp.get_unbounded_tburstpop_params(
        tbp.DEFAULT_TBURSTPOP_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        tbp.DEFAULT_TBURSTPOP_U_PARAMS._fields
    )


def test_get_bounded_tburstpop_params_fails_when_passing_params():
    try:
        tbp.get_bounded_tburstpop_params(tbp.DEFAULT_TBURSTPOP_PARAMS)
        raise NameError("get_bounded_tburstpop_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_tburstpop_params_fails_when_passing_u_params():
    try:
        tbp.get_unbounded_tburstpop_params(tbp.DEFAULT_TBURSTPOP_U_PARAMS)
        raise NameError("get_unbounded_tburstpop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_tburst_params_from_tburstpop_params_fails_when_passing_u_params():
    logsm, logssfr = 10.0, -11.0

    try:
        tbp.get_tburst_params_from_tburstpop_params(
            tbp.DEFAULT_TBURSTPOP_U_PARAMS, logsm, logssfr
        )
        raise NameError(
            "get_tburst_params_from_tburstpop_params should not accept u_params"
        )
    except AttributeError:
        pass


def test_get_tburst_params_from_tburstpop_u_params_fails_when_passing_params():
    logsm, logssfr = 10.0, -11.0

    try:
        tbp.get_tburst_params_from_tburstpop_u_params(
            tbp.DEFAULT_TBURSTPOP_PARAMS, logsm, logssfr
        )
        raise NameError(
            "get_tburst_params_from_tburstpop_u_params should not accept params"
        )
    except AttributeError:
        pass


def test_get_tburst_u_params_from_tburstpop_params_fails_when_passing_u_params():
    logsm, logssfr = 10.0, -11.0

    try:
        tbp.get_tburst_u_params_from_tburstpop_params(
            tbp.DEFAULT_TBURSTPOP_U_PARAMS, logsm, logssfr
        )
        raise NameError(
            "get_tburst_u_params_from_tburstpop_params should not accept u_params"
        )
    except AttributeError:
        pass


def test_get_tburst_u_params_from_tburstpop_u_params_fails_when_passing_params():
    logsm, logssfr = 10.0, -11.0

    try:
        tbp.get_tburst_u_params_from_tburstpop_u_params(
            tbp.DEFAULT_TBURSTPOP_PARAMS,
            logsm,
            logssfr,
        )
        raise NameError(
            "get_tburst_u_params_from_tburstpop_u_params should not accept u_params"
        )
    except AttributeError:
        pass


def test_get_tburst_u_params_from_tburstpop_params_evaluates():
    ran_key = jran.PRNGKey(0)
    ran_key, logsm_key, logssfr_key = jran.split(ran_key, 3)
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=())
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=())

    n_age = 107
    lgyr = np.linspace(5.5, 10.25, n_age)

    u_lgyr_peak, u_lgyr_max = tbp.get_tburst_u_params_from_tburstpop_params(
        tbp.DEFAULT_TBURSTPOP_PARAMS,
        logsm,
        logssfr,
    )
    assert u_lgyr_peak.shape == ()
    assert u_lgyr_max.shape == ()

    age_weights = _pureburst_age_weights_from_u_params(lgyr, u_lgyr_peak, u_lgyr_max)
    assert age_weights.shape == (n_age,)
    assert np.all(np.isfinite(age_weights))
    assert np.allclose(np.sum(age_weights), 1.0, rtol=1e-4)
    assert np.all(age_weights <= 1)
    assert np.all(age_weights >= 0)


def test_get_tburst_params_from_tburstpop_params_evaluates():
    ran_key = jran.PRNGKey(0)
    ran_key, logsm_key, logssfr_key = jran.split(ran_key, 3)
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=())
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=())

    n_age = 107
    lgyr = np.linspace(5.5, 10.25, n_age)

    lgyr_peak, lgyr_max = tbp.get_tburst_params_from_tburstpop_params(
        tbp.DEFAULT_TBURSTPOP_PARAMS,
        logsm,
        logssfr,
    )
    assert lgyr_peak.shape == ()
    assert lgyr_max.shape == ()

    age_weights = _pureburst_age_weights_from_params(lgyr, lgyr_peak, lgyr_max)
    assert age_weights.shape == (n_age,)
    assert np.all(np.isfinite(age_weights))
    assert np.allclose(np.sum(age_weights), 1.0, rtol=1e-4)
    assert np.all(age_weights <= 1)
    assert np.all(age_weights >= 0)


def test_get_tburst_params_from_tburstpop_u_params_evaluates():
    ran_key = jran.PRNGKey(0)
    ran_key, logsm_key, logssfr_key = jran.split(ran_key, 3)
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=())
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=())

    n_age = 107
    lgyr = np.linspace(5.5, 10.25, n_age)

    lgyr_peak, lgyr_max = tbp.get_tburst_params_from_tburstpop_u_params(
        tbp.DEFAULT_TBURSTPOP_U_PARAMS, logsm, logssfr
    )
    assert lgyr_peak.shape == ()
    assert lgyr_max.shape == ()

    age_weights = _pureburst_age_weights_from_params(lgyr, lgyr_peak, lgyr_max)
    assert age_weights.shape == (n_age,)
    assert np.all(np.isfinite(age_weights))
    assert np.allclose(np.sum(age_weights), 1.0, rtol=1e-4)
    assert np.all(age_weights <= 1)
    assert np.all(age_weights >= 0)


def test_get_tburst_params_from_tburstpop_u_params_are_consistent():
    ran_key = jran.PRNGKey(0)
    ran_key, logsm_key, logssfr_key = jran.split(ran_key, 3)
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=())
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=())

    n_age = 107
    lgyr = np.linspace(5.5, 10.25, n_age)

    lgyr_peak1, lgyr_max1 = tbp.get_tburst_params_from_tburstpop_u_params(
        tbp.DEFAULT_TBURSTPOP_U_PARAMS, logsm, logssfr
    )
    age_weights1 = _pureburst_age_weights_from_params(lgyr, lgyr_peak1, lgyr_max1)

    lgyr_peak2, lgyr_max2 = tbp.get_tburst_params_from_tburstpop_params(
        tbp.DEFAULT_TBURSTPOP_PARAMS, logsm, logssfr
    )
    age_weights2 = _pureburst_age_weights_from_params(lgyr, lgyr_peak2, lgyr_max2)

    assert np.allclose(lgyr_peak1, lgyr_peak2)
    assert np.allclose(lgyr_max1, lgyr_max2)
    assert np.allclose(age_weights1, age_weights2, rtol=TOL)


def test_get_tburst_u_params_from_tburstpop_u_params_are_consistent():
    ran_key = jran.PRNGKey(0)
    ran_key, logsm_key, logssfr_key = jran.split(ran_key, 3)
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=())
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=())

    n_age = 107
    lgyr = np.linspace(5.5, 10.25, n_age)

    u_lgyr_peak1, u_lgyr_max1 = tbp.get_tburst_u_params_from_tburstpop_u_params(
        tbp.DEFAULT_TBURSTPOP_U_PARAMS, logsm, logssfr
    )
    age_weights1 = _pureburst_age_weights_from_u_params(lgyr, u_lgyr_peak1, u_lgyr_max1)

    u_lgyr_peak2, u_lgyr_max2 = tbp.get_tburst_u_params_from_tburstpop_params(
        tbp.DEFAULT_TBURSTPOP_PARAMS, logsm, logssfr
    )
    age_weights2 = _pureburst_age_weights_from_u_params(lgyr, u_lgyr_peak2, u_lgyr_max2)

    assert np.allclose(u_lgyr_peak1, u_lgyr_peak2, rtol=1e-2)
    assert np.allclose(u_lgyr_max1, u_lgyr_max2, rtol=1e-2)
    assert np.allclose(age_weights1, age_weights2, rtol=1e-2)


def test_tburstpop_param_u_param_inversion():
    assert np.allclose(
        tbp.DEFAULT_TBURSTPOP_PARAMS,
        tbp.get_bounded_tburstpop_params(tbp.DEFAULT_TBURSTPOP_U_PARAMS),
        atol=0.05,
    )

    inferred_default_params = tbp.get_bounded_tburstpop_params(
        tbp.get_unbounded_tburstpop_params(tbp.DEFAULT_TBURSTPOP_PARAMS)
    )
    assert np.allclose(tbp.DEFAULT_TBURSTPOP_PARAMS, inferred_default_params, rtol=TOL)

    n_gals = 10_000
    ran_key = jran.PRNGKey(0)
    ran_key, logsm_key, logssfr_key = jran.split(ran_key, 3)
    logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=(n_gals,))
    logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=(n_gals,))

    u_lgyr_peak, u_lgyr_max = tbp.get_tburst_u_params_from_tburstpop_params(
        tbp.DEFAULT_TBURSTPOP_PARAMS, logsm, logssfr
    )
    assert u_lgyr_peak.shape == (n_gals,)
    assert u_lgyr_max.shape == (n_gals,)
    assert np.all(np.isfinite(u_lgyr_peak))
    assert np.all(np.isfinite(u_lgyr_max))

    u_lgyr_peak_u, u_lgyr_max_u = tbp.get_tburst_u_params_from_tburstpop_u_params(
        tbp.DEFAULT_TBURSTPOP_U_PARAMS, logsm, logssfr
    )
    assert u_lgyr_peak_u.shape == (n_gals,)
    assert u_lgyr_max_u.shape == (n_gals,)
    assert np.all(np.isfinite(u_lgyr_peak_u))
    assert np.all(np.isfinite(u_lgyr_max_u))

    assert np.allclose(u_lgyr_max, u_lgyr_max_u, atol=0.05)
    assert np.allclose(u_lgyr_peak, u_lgyr_peak_u, atol=0.05)
