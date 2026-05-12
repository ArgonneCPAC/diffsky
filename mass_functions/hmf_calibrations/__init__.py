""" """

# flake8: noqa


import os

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))

from .hmf_param_reader import load_hmf_model_params_from_txt
from .smdpl_hmf import HMF_PARAMS as DEFAULT_HMF_PARAMS  # noqa
from .smdpl_hmf import HMF_Params  # noqa

LJ_HMF_PARAMS = load_hmf_model_params_from_txt(
    os.path.join(_THIS_DRNAME, "lj_hmf_params.txt")
)
DISC_LCDM_HMF_PARAMS = load_hmf_model_params_from_txt(
    os.path.join(_THIS_DRNAME, "disc_lcdm_hmf_params.txt")
)
