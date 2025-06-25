""" """

import os

from ....hmf_calibrations import smdpl_hmf, smdpl_hmf_fitting_helpers
from .. import hmf_fit_diagnostics

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
ROOT_DRN = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DRNAME)))
DRN_TESTING_DATA = os.path.join(ROOT_DRN, "hmf_calibrations", "tests", "testing_data")


def test_make_hmf_fit_plot():
    loss_data = smdpl_hmf_fitting_helpers.get_loss_data(DRN_TESTING_DATA, "hosthalos")
    p_best = smdpl_hmf.HMF_PARAMS
    hmf_fit_diagnostics.make_hmf_fit_plot(loss_data, p_best)
