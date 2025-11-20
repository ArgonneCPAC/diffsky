""""""

import numpy as np

from .. import diffsky_param_wrapper as dpw
from .. import param_loader


def test_load_diffsky_param_collection():
    all_params_flat = dpw.unroll_param_collection_into_flat_array(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    all_pnames = dpw.get_flat_param_names()
    dtype = [
        (name, np.array(val).dtype, np.array(val).shape)
        for name, val in zip(all_pnames, all_params_flat)
    ]
    all_params_flat_structured = np.array([all_params_flat], dtype=dtype)

    np.save("all_params_flat", all_params_flat_structured)
    param_collection = param_loader.load_diffsky_param_collection("all_params_flat.npy")
    all_params_flat2 = dpw.unroll_param_collection_into_flat_array(*param_collection)

    assert np.allclose(all_params_flat, all_params_flat2, rtol=1e-5)
