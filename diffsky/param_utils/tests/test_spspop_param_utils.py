"""
"""

from ..spspop_param_utils import (
    DEFAULT_SPSPOP_PARAMS,
    DEFAULT_SPSPOP_U_PARAMS,
    get_bounded_spspop_params_tw_dust,
    get_unbounded_spspop_params_tw_dust,

)


def test_spspop_param_u_param_names_propagate_properly():
    gen = zip(DEFAULT_SPSPOP_U_PARAMS._fields, DEFAULT_SPSPOP_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = get_bounded_spspop_params_tw_dust(DEFAULT_SPSPOP_U_PARAMS)
    assert set(inferred_default_params._fields) == set(DEFAULT_SPSPOP_PARAMS._fields)

    inferred_default_u_params = get_unbounded_spspop_params_tw_dust(DEFAULT_SPSPOP_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        DEFAULT_SPSPOP_U_PARAMS._fields
    )


def test_get_bounded_spspop_params_fails_when_passing_params():
    try:
        get_bounded_spspop_params_tw_dust(DEFAULT_SPSPOP_PARAMS)
        raise NameError("get_bounded_spspop_params should not accept u_params")
    except AttributeError:
        pass


def test_get_unbounded_spspop_params_fails_when_passing_u_params():
    try:
        get_unbounded_spspop_params_tw_dust(DEFAULT_SPSPOP_U_PARAMS)
        raise NameError("get_unbounded_scatter_params should not accept u_params")
    except AttributeError:
        pass
