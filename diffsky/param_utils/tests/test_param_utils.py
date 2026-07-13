""""""

from .. import DIFFSKY_FIT_PARAMS, FENIKS_FIT_PARAMS
from .. import diffsky_param_wrapper_merging as dpwm


def test_diffsky_fit_params_are_ok():
    for calib_name, calib_params in DIFFSKY_FIT_PARAMS.items():
        msg = f"`{calib_name}` fails check_param_collection_is_ok"
        assert dpwm.check_param_collection_is_ok(calib_params), msg


def test_feniks_fit_params_are_ok():
    for calib_name, calib_params in FENIKS_FIT_PARAMS.items():
        msg = f"`{calib_name}` fails check_param_collection_is_ok"
        assert dpwm.check_param_collection_is_ok(calib_params), msg
