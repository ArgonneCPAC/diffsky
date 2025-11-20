""""""

from collections import namedtuple

import numpy as np

from ...data_loaders import io_utils as iou
from .. import diffsky_param_wrapper as dpw
from .. import param_loader


def test_load_diffsky_param_collection():
    all_params_flat = dpw.unroll_param_collection_into_flat_array(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    all_pnames = dpw.get_flat_param_names()

    Params = namedtuple("Params", all_pnames)
    all_named_params = Params(*all_params_flat)

    fn = "diffsky_unit_testing_params.hdf5"
    iou.write_namedtuple_to_hdf5(all_named_params, fn)

    param_collection = param_loader.load_diffsky_param_collection(fn)
    all_params_flat2 = dpw.unroll_param_collection_into_flat_array(*param_collection)

    assert np.allclose(all_params_flat, all_params_flat2, rtol=1e-5)
