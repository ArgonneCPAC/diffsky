""""""

from collections import namedtuple

from ..data_loaders import io_utils as iou
from . import diffsky_param_wrapper as dpw


def load_diffsky_param_collection(fn):
    """"""
    flat_diffsky_params = iou.load_namedtuple_from_hdf5(fn)
    param_collection = dpw.get_param_collection_from_flat_array(flat_diffsky_params)
    return param_collection


def write_diffsky_param_collection(fn_out, param_collection):
    """"""
    flat_diffsky_params = dpw.unroll_param_collection_into_flat_array(*param_collection)
    DiffskyParams = namedtuple("DiffskyParams", dpw.get_flat_param_names())
    flat_diffsky_params = DiffskyParams(*flat_diffsky_params)

    iou.write_namedtuple_to_hdf5(flat_diffsky_params, fn_out)
