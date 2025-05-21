""" """

from .. import diffsky_param_wrapper as dpw


def test_get_flat_param_names():
    flat_param_names = dpw.get_flat_param_names()
    assert len(flat_param_names) == len(set(flat_param_names))


def test_get_param_collection_from_u_param_array():

    param_collection = dpw.unroll_param_collection_into_flat_array(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    u_param_collection = dpw.get_u_param_collection_from_param_collection(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    u_param_arr = dpw.unroll_u_param_collection_into_flat_array(*u_param_collection)
    u_param_collection2 = dpw.get_u_param_collection_from_u_param_array(u_param_arr)
