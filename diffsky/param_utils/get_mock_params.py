""""""

from ..experimental.sfh_model_calibrations import (
    load_diffsky_sfh_model_calibrations as ldup,
)
from . import diffsky_param_wrapper as dpw
from .cosmos_calibrations import COSMOS_PARAM_FITS


def get_param_collection_for_mock(cosmos_fit="", sfh_model="", rank=0):
    """Load diffsky model parameters"""
    if cosmos_fit != "":
        if rank == 0:
            print(f"Using cosmos_fit = {cosmos_fit}")
        param_collection = COSMOS_PARAM_FITS[cosmos_fit]
    elif sfh_model != "":
        if rank == 0:
            print(f"Loading diffsky model parameters for sfh_model={sfh_model}")
        u_param_arr = ldup.load_diffsky_u_params_for_sfh_model(sfh_model)
        u_param_collection = dpw.get_u_param_collection_from_u_param_array(u_param_arr)
        param_collection = dpw.get_param_collection_from_u_param_collection(
            *u_param_collection
        )
    else:
        param_collection = dpw.DEFAULT_PARAM_COLLECTION
        if rank == 0:
            print(
                "No input params detected. "
                "Using default diffsky model parameters DEFAULT_PARAM_COLLECTION"
            )
    return param_collection
