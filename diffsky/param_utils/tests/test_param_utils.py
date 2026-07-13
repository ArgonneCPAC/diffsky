""""""

from .. import COSMOS_FIT_PARAMS
from .. import diffsky_param_wrapper_merging as dpwm


def test_all_calibrations_are_ok():
    for calib_name, calib_params in COSMOS_FIT_PARAMS.items():
        msg = f"`{calib_name}` fails check_param_collection_is_ok"
        assert dpwm.check_param_collection_is_ok(calib_params), msg
