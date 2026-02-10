""""""

from ...experimental.sfh_model_calibrations import (
    load_diffsky_sfh_model_calibrations as ldup,
)
from .. import COSMOS_PARAM_FITS
from .. import diffsky_param_wrapper as dpw
from .. import get_mock_params as gmp

KNOWN_COSMOS_CALIBRATION_FAILURES = ["cosmos_260120_UM", "cosmos_260105"]


def test_get_param_collection_for_mock_sfh_model_options():
    for sfh_model in ldup.DIFFSTARPOP_CALIBRATIONS:
        param_collection = gmp.get_param_collection_for_mock(
            sfh_model=sfh_model, rank=0
        )
        assert param_collection._fields == dpw.ParamCollection._fields


def test_get_param_collection_for_mock_cosmos_fit_options():
    for key in COSMOS_PARAM_FITS.keys():
        param_collection = gmp.get_param_collection_for_mock(cosmos_fit=key, rank=0)
        assert param_collection._fields == dpw.ParamCollection._fields


def test_get_param_collection_for_mock_defaults():
    param_collection = gmp.get_param_collection_for_mock()
    assert param_collection._fields == dpw.ParamCollection._fields


def test_cosmos_param_fits_are_ok():
    """Enforce all parameters in COSMOS_PARAM_FITS pass sanity checks encoded in
    dpw.check_param_collection_is_ok function

    Calibrations appearing in list KNOWN_COSMOS_CALIBRATION_FAILURES are permitted
    to fail. So this test only checks for new failures beyond this list.

    """
    bad_params = []
    for key in COSMOS_PARAM_FITS.keys():
        param_collection = gmp.get_param_collection_for_mock(cosmos_fit=key, rank=0)
        param_collection_is_ok = dpw.check_param_collection_is_ok(param_collection)
        if not param_collection_is_ok:
            bad_params.append(key)

    bad_params = list(set(bad_params) - set(KNOWN_COSMOS_CALIBRATION_FAILURES))

    msg = "Bounding/unbounding failures of some params in dpw.COSMOS_PARAM_FITS\n"
    if len(bad_params) > 0:
        print(msg)
        for bad_param in bad_params:
            print(f"{bad_param} fails")
        assert param_collection_is_ok, f"{key} fails check_param_collection_is_ok"
