""" """

import os

import numpy as np
from diffstar.diffstarpop.kernels.params.params_diffstarpopfits_mgash import (
    DiffstarPop_UParams_Diffstarpopfits_mgash as diffstarpop_models_u_p_dict,
)

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def load_diffsky_u_params_for_sfh_model(sfh_model_nickname, bnpat="u_params_best_1015"):
    drn = os.path.join(_THIS_DRNAME, "data")
    bn = f"{bnpat}_{sfh_model_nickname}.npy"
    fn = os.path.join(drn, bn)
    u_params_sps = np.load(fn)
    u_params_sfh = diffstarpop_models_u_p_dict[sfh_model_nickname]
    u_params = np.concatenate((u_params_sfh, u_params_sps))
    return u_params
