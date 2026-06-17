""""""

from .. import diffsky_param_wrapper_merging as dpwm
from .. import load_calib_params


def test_get_calib_params():
    param_collection = load_calib_params.get_calib_params()
    assert dpwm.check_param_collection_is_ok(param_collection)
