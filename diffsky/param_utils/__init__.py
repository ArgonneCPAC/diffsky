""""""

import os

from .load_calib_params import get_calib_params

COSMOS_FIT_PARAMS = dict()
COSMOS_FIT_PARAMS["c260710"] = get_calib_params(
    calibration_dir=os.path.join("cosmos_calibrations", "data"),
    calibration_name="c260710",
)

FENIKS_FIT_PARAMS = dict()
FENIKS_FIT_PARAMS["feniks_260617"] = get_calib_params(
    calibration_dir=os.path.join("feniks_calibrations"),
    calibration_name="feniks_260617",
)
FENIKS_FIT_PARAMS["sdss_feniks_260701"] = get_calib_params(
    calibration_dir=os.path.join("sdss_feniks_calibrations"),
    calibration_name="sdss_feniks_260701",
)

FENIKS_FIT_PARAMS["sdss_feniks_hizels_260710"] = get_calib_params(
    calibration_dir=os.path.join("sdss_feniks_hizels_calibrations"),
    calibration_name="sdss_feniks_hizels_260710",
)


DIFFSKY_FIT_PARAMS = dict()
DIFFSKY_FIT_PARAMS.update(COSMOS_FIT_PARAMS)
DIFFSKY_FIT_PARAMS.update(FENIKS_FIT_PARAMS)
