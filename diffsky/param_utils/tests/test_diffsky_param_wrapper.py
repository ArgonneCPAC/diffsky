""" """

from .. import diffsky_param_wrapper as dpw


def test_get_flat_param_names():
    flat_param_names = dpw.get_flat_param_names()
    assert len(flat_param_names) == len(set(flat_param_names))
