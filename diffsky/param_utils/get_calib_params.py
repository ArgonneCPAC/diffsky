import os

import diffsky

from ..data_loaders.hacc_utils import lc_mock


def get_calib_params(
    calibration_dir="feniks_calibrations", calibration_name="feniks_260617"
):
    diffsky_dir = os.path.dirname(diffsky.__file__)

    calib_dir = os.path.join(diffsky_dir, "param_utils", calibration_dir)

    param_collection = lc_mock.load_diffsky_param_collection_merging(
        calib_dir,
        calibration_name,
    )

    return param_collection
