"""
"""

import os
from glob import glob

import numpy as np

from ..hmf_model import DEFAULT_HMF_PARAMS, predict_cuml_hmf

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TESTING_DATA_DRN = os.path.join(_THIS_DRNAME, "testing_data")
BNPAT = "smdpl_hmf_cuml_redshift_{0:.2f}.txt"


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
    fname_list = glob(os.path.join(TESTING_DATA_DRN, "smdpl_hmf_*.txt"))

    for fn in fname_list:
        arr = np.loadtxt(fn)
        lgmp_target, hmf_target = arr[:, 0], arr[:, 1]
        bn = os.path.basename(fn)
        redshift = infer_redshift_from_bname(bn)

        hmf_pred = predict_cuml_hmf(DEFAULT_HMF_PARAMS, lgmp_target, redshift)

        assert _mse(hmf_pred, hmf_target) < 0.01
