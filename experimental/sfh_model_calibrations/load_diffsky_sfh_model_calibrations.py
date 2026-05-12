""" """

import os

import numpy as np
from diffstar.diffstarpop.kernels.params.params_diffstarpopfits_mgash import (
    DiffstarPop_UParams_Diffstarpopfits_mgash as diffstarpop_models_u_p_dict,
)

from ...param_utils import diffsky_param_wrapper as dpw

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DIFFSTARPOP_CALIBRATIONS = [
    "smdpl_dr1_nomerging",
    "smdpl_dr1",
    "tng",
    "galacticus_in_plus_ex_situ",
    "galacticus_in_situ",
]
SPSOP_CALIBRATIONS = ["smdpl_dr1", "tng", "galacticus_in_plus_ex_situ"]


def load_diffsky_u_params_for_sfh_model(
    sfh_model_nickname, bnpat="u_params_best_1015", rank=0
):
    """"""
    try:
        u_params_sfh = diffstarpop_models_u_p_dict[sfh_model_nickname]
    except KeyError:
        msg = (
            f"sfh_model_nickname={sfh_model_nickname} not recognized. "
            "Options are {DIFFSTARPOP_CALIBRATIONS}"
        )
        raise KeyError(msg)

    if sfh_model_nickname in SPSOP_CALIBRATIONS:
        drn = os.path.join(_THIS_DRNAME, "data")
        bn = f"{bnpat}_{sfh_model_nickname}.npy"
        fn = os.path.join(drn, bn)
        u_params_sps = np.load(fn)  # noqa

        if True:  # always do this for now to ignore the u_params_sps calbrations
            if rank == 0:
                print("Ignoring pre-computed fit of SPS parameters")

            u_param_collection = dpw.get_u_param_collection_from_param_collection(
                *dpw.DEFAULT_PARAM_COLLECTION
            )
            u_params = dpw.unroll_u_param_collection_into_flat_array(
                *u_param_collection
            )
            n_sfh_params = len(u_params_sfh)
            u_params = np.concatenate((u_params_sfh, u_params[n_sfh_params:]))

    elif sfh_model_nickname in DIFFSTARPOP_CALIBRATIONS:
        param_collection = dpw.DEFAULT_PARAM_COLLECTION
        u_param_collection = dpw.get_u_param_collection_from_param_collection(
            *param_collection
        )
        u_param_collection = u_param_collection._replace(
            diffstarpop_u_params=u_params_sfh
        )
        u_params = dpw.unroll_u_param_collection_into_flat_array(*u_param_collection)
    else:
        raise ValueError(f"Unrecognized `sfh_model_nickname`={sfh_model_nickname}")

    return u_params
