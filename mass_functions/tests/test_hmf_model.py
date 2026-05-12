""" """

import os

import numpy as np

from ..fitting_utils import fit_hmf_model
from ..hmf_calibrations.smdpl_hmf_fitting_helpers import get_loss_data
from ..hmf_model import DEFAULT_HMF_PARAMS, predict_cuml_hmf

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TESTING_DATA_DRN = os.path.join(
    _THIS_DRNAME, "hmf_calibrations", "tests", "testing_data"
)


def infer_redshift_from_bname(bn):
    return float(bn.split("_")[-1][:-4])


def test_lg_hmf_kern_evaluates():
    lgmp_arr = np.linspace(10, 15, 500)
    redshift = 0.0
    hmf = predict_cuml_hmf(DEFAULT_HMF_PARAMS, lgmp_arr, redshift)
    assert hmf.shape == hmf.shape
    assert np.all(np.isfinite(hmf))


def _mse(pred, target):
    diff = pred - target
    return np.mean(diff**2)


def test_predict_hmf_returns_finite_valued_expected_shape():
    redshift = 1.0
    nhalos = 100
    lgmp_arr = np.linspace(10, 15, nhalos)
    pred = predict_cuml_hmf(DEFAULT_HMF_PARAMS, lgmp_arr, redshift)
    assert pred.shape == lgmp_arr.shape
    assert np.all(np.isfinite(pred))

    lgmp = 12.0
    pred = predict_cuml_hmf(DEFAULT_HMF_PARAMS, lgmp, redshift)
    assert pred.shape == ()
    assert np.all(np.isfinite(pred))

    nhalos = 5
    zarr = np.linspace(0, 5, nhalos)
    lgmp = 12.0
    pred = predict_cuml_hmf(DEFAULT_HMF_PARAMS, lgmp, zarr)
    assert pred.shape == (nhalos,)
    assert np.all(np.isfinite(pred))

    nhalos = 5
    zarr = np.linspace(0, 5, nhalos)
    lgmp_arr = np.linspace(10, 15, nhalos)
    pred = predict_cuml_hmf(DEFAULT_HMF_PARAMS, lgmp_arr, zarr)
    assert pred.shape == (nhalos,)
    assert np.all(np.isfinite(pred))


def test_predict_hmf_accurately_approximates_simulation_data():
    """This test loads some pretabulated HMF data computed from SMPDL
    and compares the simulation results to the predict_hmf function"""

    loss_data = get_loss_data(TESTING_DATA_DRN, "hosthalos")
    loss = fit_hmf_model._loss_func_multi_z(DEFAULT_HMF_PARAMS, loss_data)
    assert loss < 0.01
