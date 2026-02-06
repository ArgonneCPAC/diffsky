""" """

# flake8: noqa
from copy import deepcopy

import numpy as np

from .. import diffsky_param_wrapper as dpw


def test_get_flat_param_names():
    flat_param_names = dpw.get_flat_param_names()
    assert len(flat_param_names) == len(set(flat_param_names))


def test_default_param_collection_fields():
    pnames = (
        "diffstarpop_params",
        "mzr_params",
        "spspop_params",
        "scatter_params",
        "ssperr_params",
    )
    assert dpw.DEFAULT_PARAM_COLLECTION._fields == pnames


def test_get_param_collection_from_flat_array():
    """Enforce agreement when we roll up a flat array and then repack it"""
    default_param_collection = deepcopy(dpw.DEFAULT_PARAM_COLLECTION)

    all_params_flat = dpw.unroll_param_collection_into_flat_array(
        *default_param_collection
    )
    param_collection = dpw.get_param_collection_from_flat_array(all_params_flat)

    assert np.allclose(param_collection[0], default_param_collection[0])
    assert np.allclose(param_collection[1], default_param_collection[1])
    assert np.allclose(
        param_collection[2].burstpop_params.freqburst_params,
        default_param_collection[2].burstpop_params.freqburst_params,
    )
    assert np.allclose(
        param_collection[2].burstpop_params.fburstpop_params,
        default_param_collection[2].burstpop_params.fburstpop_params,
    )
    assert np.allclose(
        param_collection[2].burstpop_params.tburstpop_params,
        default_param_collection[2].burstpop_params.tburstpop_params,
    )

    assert np.allclose(
        param_collection[2].dustpop_params.avpop_params,
        default_param_collection[2].dustpop_params.avpop_params,
    )
    assert np.allclose(
        param_collection[2].dustpop_params.deltapop_params,
        default_param_collection[2].dustpop_params.deltapop_params,
    )
    assert np.allclose(
        param_collection[2].dustpop_params.funopop_params,
        default_param_collection[2].dustpop_params.funopop_params,
    )

    assert np.allclose(param_collection[3], default_param_collection[3])
    assert np.allclose(param_collection[4], default_param_collection[4])


def test_unroll_u_param_collection_into_flat_array():

    u_param_collection = dpw.get_u_param_collection_from_param_collection(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    u_param_arr = dpw.unroll_u_param_collection_into_flat_array(*u_param_collection)
    u_param_collection2 = dpw.get_u_param_collection_from_u_param_array(u_param_arr)

    u_param_arr2 = dpw.unroll_u_param_collection_into_flat_array(*u_param_collection2)

    assert np.allclose(u_param_arr, u_param_arr2, rtol=1e-4)


def test_get_param_collection_from_u_param_collection():
    u_param_collection = dpw.get_u_param_collection_from_param_collection(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    param_collection = dpw.get_param_collection_from_u_param_collection(
        *u_param_collection
    )

    u_param_collection2 = dpw.get_u_param_collection_from_param_collection(
        *param_collection
    )

    assert np.allclose(u_param_collection[0], u_param_collection2[0])
    assert np.allclose(u_param_collection[1], u_param_collection2[1])

    assert np.allclose(
        u_param_collection[2].u_burstpop_params.freqburst_u_params,
        u_param_collection2[2].u_burstpop_params.freqburst_u_params,
    )
    assert np.allclose(
        u_param_collection[2].u_burstpop_params.fburstpop_u_params,
        u_param_collection2[2].u_burstpop_params.fburstpop_u_params,
    )
    assert np.allclose(
        u_param_collection[2].u_burstpop_params.tburstpop_u_params,
        u_param_collection2[2].u_burstpop_params.tburstpop_u_params,
    )

    assert np.allclose(
        u_param_collection[2].u_dustpop_params.avpop_u_params,
        u_param_collection2[2].u_dustpop_params.avpop_u_params,
    )
    assert np.allclose(
        u_param_collection[2].u_dustpop_params.deltapop_u_params,
        u_param_collection2[2].u_dustpop_params.deltapop_u_params,
    )
    assert np.allclose(
        u_param_collection[2].u_dustpop_params.funopop_u_params,
        u_param_collection2[2].u_dustpop_params.funopop_u_params,
    )

    assert np.allclose(u_param_collection[3], u_param_collection2[3])
    assert np.allclose(u_param_collection[4], u_param_collection2[4])


def test_unroll_param_collection_into_flat_array():
    param_arr = dpw.unroll_param_collection_into_flat_array(
        *dpw.DEFAULT_PARAM_COLLECTION
    )

    u_param_collection = dpw.get_u_param_collection_from_param_collection(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    param_collection = dpw.get_param_collection_from_u_param_collection(
        *u_param_collection
    )
    param_arr2 = dpw.unroll_param_collection_into_flat_array(*param_collection)

    assert np.allclose(param_arr, param_arr2, rtol=1e-4)


def test_default_diffsky_params_are_ok():
    param_collection_is_ok = dpw.check_param_collection_is_ok(
        dpw.DEFAULT_PARAM_COLLECTION
    )
    assert param_collection_is_ok
