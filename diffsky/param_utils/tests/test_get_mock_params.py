""""""

from ...experimental.sfh_model_calibrations import (
    load_diffsky_sfh_model_calibrations as ldup,
)
from .. import diffsky_param_wrapper as dpw
from .. import get_mock_params as gmp


def test_get_param_collection_for_mock_sfh_model_options():
    for sfh_model in ldup.DIFFSTARPOP_CALIBRATIONS:
        param_collection = gmp.get_param_collection_for_mock(
            sfh_model=sfh_model, rank=0
        )
        assert param_collection._fields == dpw.ParamCollection._fields


def test_get_param_collection_for_mock_cosmos_fit_options():
    param_collection = gmp.get_param_collection_for_mock(
        cosmos_fit="cosmos260105", rank=0
    )
    assert param_collection._fields == dpw.ParamCollection._fields


def test_get_param_collection_for_mock_defaults():
    param_collection = gmp.get_param_collection_for_mock()
    assert param_collection._fields == dpw.ParamCollection._fields
