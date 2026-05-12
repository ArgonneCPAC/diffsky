""""""

import os

from ...fitting_utils import fit_hmf_model
from .. import smdpl_hmf, smdpl_hmf_subs
from ..smdpl_hmf_fitting_helpers import get_loss_data

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DRN_TESTING_DATA = os.path.join(_THIS_DRNAME, "testing_data")

HMF_FIT_TOL = 0.02


def test_calibrated_hmf_model_accuracy_hosthalos():
    loss_data = get_loss_data(DRN_TESTING_DATA, "hosthalos")
    loss_best = fit_hmf_model._loss_func_multi_z(smdpl_hmf.HMF_PARAMS, loss_data)

    assert loss_best < HMF_FIT_TOL


def test_calibrated_hmf_model_accuracy_subhalos():
    loss_data = get_loss_data(DRN_TESTING_DATA, "subhalos")
    loss_best = fit_hmf_model._loss_func_multi_z(smdpl_hmf_subs.HMF_PARAMS, loss_data)

    assert loss_best < HMF_FIT_TOL


def test_hmf_model_accuracy_tol_is_small_enough_to_be_meaningful():
    """If we use the subhalo fit for host halo target data, the test should fail"""
    loss_data = get_loss_data(DRN_TESTING_DATA, "subhalos")
    loss_best = fit_hmf_model._loss_func_multi_z(smdpl_hmf.HMF_PARAMS, loss_data)
    assert loss_best > 2 * HMF_FIT_TOL


def test_hmf_fitter_successfull_minimizes_loss():
    """Refit the HMF of subhalos using the host halo calibration as starting point.
    Verify that the loss indeed improves"""
    loss_data_subs = get_loss_data(DRN_TESTING_DATA, "subhalos")
    loss_init = fit_hmf_model._loss_func_multi_z(smdpl_hmf.HMF_PARAMS, loss_data_subs)
    assert loss_init > HMF_FIT_TOL

    _res = fit_hmf_model.hmf_fitter(loss_data_subs, p_init=smdpl_hmf.HMF_PARAMS)
    p_best = _res[0]
    loss_best = fit_hmf_model._loss_func_multi_z(p_best, loss_data_subs)
    assert loss_best < HMF_FIT_TOL < loss_init

    fit_terminates = _res[4]
    assert fit_terminates == 1
