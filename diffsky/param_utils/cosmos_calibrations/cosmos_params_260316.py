""""""

import os
from importlib.resources import files

from ...data_loaders import io_utils as iou
from .. import diffsky_param_wrapper_merging as dpwm

BNAME_PARAMS = "diffsky_cosmos_260316_mrg_param_collection.hdf5"


def _load_param_collection_from_flat_hdf5():
    drn = files("diffsky.param_utils.cosmos_calibrations") / "data"
    fn = os.path.join(drn, BNAME_PARAMS)
    flat_diffsky_params = iou.load_namedtuple_from_hdf5(fn)
    param_collection = dpwm.get_param_collection_from_flat_array(flat_diffsky_params)
    return param_collection


cosmos_260316 = _load_param_collection_from_flat_hdf5()
