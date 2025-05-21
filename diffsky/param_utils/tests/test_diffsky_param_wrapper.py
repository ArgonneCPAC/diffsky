""" """

# flake8: noqa

import numpy as np

from .. import diffsky_param_wrapper as dpw


def test_get_flat_param_names():
    flat_param_names = dpw.get_flat_param_names()
    assert len(flat_param_names) == len(set(flat_param_names))


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
    assert np.allclose(
        u_param_collection[0].u_sfh_pdf_cens_params,
        u_param_collection2[0].u_sfh_pdf_cens_params,
    )

    assert np.allclose(
        u_param_collection[0].u_satquench_params,
        u_param_collection2[0].u_satquench_params,
    )

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
