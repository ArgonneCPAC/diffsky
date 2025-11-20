""""""

from collections import namedtuple

import numpy as np

from . import diffsky_param_wrapper as dpw


def load_diffsky_param_collection(fn):
    diffsky_params_str_arr = np.load(fn)

    param_names = dpw.get_flat_param_names()
    DiffskyParams = namedtuple("DiffskyParams", param_names)

    reloaded_params = DiffskyParams(
        *[diffsky_params_str_arr[key] for key in DiffskyParams._fields]
    )

    param_collection = dpw.get_param_collection_from_flat_array(reloaded_params)

    return param_collection
