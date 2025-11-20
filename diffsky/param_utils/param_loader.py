""""""

from ..data_loaders import io_utils as iou
from . import diffsky_param_wrapper as dpw


def load_diffsky_param_collection(fn):
    """"""
    flat_diffsky_params = iou.load_namedtuple_from_hdf5(fn)
    param_collection = dpw.get_param_collection_from_flat_array(flat_diffsky_params)
    return param_collection
