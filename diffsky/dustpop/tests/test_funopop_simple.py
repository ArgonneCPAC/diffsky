"""
"""
import numpy as np

from ..funopop_simple import (
    DEFAULT_FUNOPOP_PARAMS,
    DEFAULT_FUNOPOP_U_PARAMS,
    get_bounded_funopop_params,
    get_funo_from_funopop_params,
    get_funo_from_funopop_u_params,
    get_unbounded_funopop_params,
)

TOL = 1e-2


def test_param_u_param_names_propagate_properly():
    gen = zip(DEFAULT_FUNOPOP_U_PARAMS._fields, DEFAULT_FUNOPOP_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = get_bounded_funopop_params(DEFAULT_FUNOPOP_U_PARAMS)
    assert set(inferred_default_params._fields) == set(DEFAULT_FUNOPOP_PARAMS._fields)

    inferred_default_u_params = get_unbounded_funopop_params(DEFAULT_FUNOPOP_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        DEFAULT_FUNOPOP_U_PARAMS._fields
    )


def test_get_bounded_funopop_params_fails_when_passing_params():
    try:
        get_bounded_funopop_params(DEFAULT_FUNOPOP_PARAMS)
        raise NameError("get_bounded_funopop_params should not accept params")
    except AttributeError:
        pass


def test_get_unbounded_funopop_params_fails_when_passing_u_params():
    try:
        get_unbounded_funopop_params(DEFAULT_FUNOPOP_U_PARAMS)
        raise NameError("get_unbounded_funopop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_funo_from_funopop_params_fails_when_passing_u_params():
    logsm, logssfr = 10.0, -10.5

    try:
        get_funo_from_funopop_params(DEFAULT_FUNOPOP_U_PARAMS, logsm, logssfr)
        raise NameError("get_funo_from_funopop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_funo_from_funopop_u_params_fails_when_passing_params():
    logsm, logssfr = 10.0, -10.5

    try:
        get_funo_from_funopop_u_params(DEFAULT_FUNOPOP_PARAMS, logsm, logssfr)
        raise NameError("get_funo_from_funopop_u_params should not accept params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    p = get_bounded_funopop_params(DEFAULT_FUNOPOP_U_PARAMS)
    u_p = get_unbounded_funopop_params(p)
    assert np.all(np.isfinite(u_p))
    assert np.allclose(u_p, DEFAULT_FUNOPOP_U_PARAMS, rtol=TOL)

    assert np.allclose(DEFAULT_FUNOPOP_PARAMS, get_bounded_funopop_params(u_p))


def test_get_funo_from_u_params_singlegal():
    logsm, logssfr = 10.0, -10.5

    funo = get_funo_from_funopop_u_params(DEFAULT_FUNOPOP_U_PARAMS, logsm, logssfr)
    assert funo.shape == ()
    assert np.all(funo >= 0)
    assert np.all(funo <= 1)

    funopop_params = get_bounded_funopop_params(DEFAULT_FUNOPOP_U_PARAMS)
    funo2 = get_funo_from_funopop_params(funopop_params, logsm, logssfr)
    assert np.allclose(funo, funo2, rtol=TOL)


def test_get_funo_from_u_params_galpop():
    logsm, logssfr = 10.0, -10.5

    n_gals = 255
    zz = np.zeros(n_gals)
    funo = get_funo_from_funopop_u_params(
        DEFAULT_FUNOPOP_U_PARAMS, logsm + zz, logssfr + zz
    )
    assert funo.shape == (n_gals,)
    assert np.all(funo >= 0)
    assert np.all(funo <= 1)

    funopop_params = get_bounded_funopop_params(DEFAULT_FUNOPOP_U_PARAMS)
    funo2 = get_funo_from_funopop_params(funopop_params, logsm + zz, logssfr + zz)
    assert np.allclose(funo, funo2, rtol=TOL)
