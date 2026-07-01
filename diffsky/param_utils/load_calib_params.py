import os

from ..data_loaders.io_utils import load_namedtuple_from_hdf5
from . import diffsky_param_wrapper_merging as dpwm

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))

BNPAT_PARAM_COLLECTION = "diffsky_{0}_param_collection.hdf5"


def get_calib_params(
    calibration_dir="feniks_calibrations", calibration_name="feniks_260617"
):
    drn = os.path.join(_THIS_DRNAME, calibration_dir)
    bname = BNPAT_PARAM_COLLECTION.format(calibration_name)
    fname = os.path.join(drn, bname)

    param_collection = load_param_collection(fname)
    return param_collection


def load_param_collection(fn):
    """"""
    flat_diffsky_params = load_namedtuple_from_hdf5(fn)
    param_collection = dpwm.get_param_collection_from_flat_array(flat_diffsky_params)
    return param_collection
