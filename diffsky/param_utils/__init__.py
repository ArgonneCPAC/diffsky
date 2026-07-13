""""""

import os

from .cosmos_calibrations import COSMOS_PARAM_FITS, COSMOS_PARAM_FITS_MERGING  # noqa
from .load_calib_params import get_calib_params

COSMOS_FIT_PARAMS = dict()
COSMOS_FIT_PARAMS["c260710"] = get_calib_params(
    calibration_dir=os.path.join("cosmos_calibrations", "data"),
    calibration_name="c260710",
)
