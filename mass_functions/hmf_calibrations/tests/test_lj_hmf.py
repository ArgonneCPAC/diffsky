""""""

import os

import numpy as np

from ... import hmf_model
from ...fitting_utils import fit_hmf_model
from .. import LJ_HMF_PARAMS

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DRN_TESTING_DATA = os.path.join(_THIS_DRNAME, "testing_data")

HMF_FIT_TOL = 0.05


def test_lj_hmf_model_loss_is_small():
    lj_hmf_redshift_bins = np.loadtxt(
        os.path.join(DRN_TESTING_DATA, "lj_hmf_redshift_bins.txt")
    )
    lj_hmf_logmp_bins = np.loadtxt(
        os.path.join(DRN_TESTING_DATA, "lj_hmf_logmp_bins.txt")
    )
    lj_hmf_cuml_density = np.loadtxt(
        os.path.join(DRN_TESTING_DATA, "lj_hmf_cuml_density_cens.txt")
    )

    loss_data_collector = []
    for iz, redshift in enumerate(lj_hmf_redshift_bins):
        msk_logmp = lj_hmf_cuml_density[:, iz] > 0
        logmp_target = lj_hmf_logmp_bins[msk_logmp]
        lj_hmf_cuml_density_target = lj_hmf_cuml_density[:, iz][msk_logmp]
        loss_data = redshift, logmp_target, np.log10(lj_hmf_cuml_density_target)
        loss_data_collector.append(loss_data)

    loss_default = fit_hmf_model._loss_func_multi_z(LJ_HMF_PARAMS, loss_data_collector)
    assert loss_default < HMF_FIT_TOL


def test_lj_hmf_model_recomputed_loss_is_small():

    lj_hmf_redshift_bins = np.loadtxt(
        os.path.join(DRN_TESTING_DATA, "lj_hmf_redshift_bins.txt")
    )
    lj_hmf_logmp_bins = np.loadtxt(
        os.path.join(DRN_TESTING_DATA, "lj_hmf_logmp_bins.txt")
    )
    lj_hmf_cuml_density = np.loadtxt(
        os.path.join(DRN_TESTING_DATA, "lj_hmf_cuml_density_cens.txt")
    )
    for iz, redshift in enumerate(lj_hmf_redshift_bins):
        msk_logmp = lj_hmf_cuml_density[:, iz] > 0
        logmp_target = lj_hmf_logmp_bins[msk_logmp]
        lj_hmf_cuml_density_target = np.log10(lj_hmf_cuml_density[:, iz][msk_logmp])

        lg_cuml_density_pred = hmf_model.predict_cuml_hmf(
            LJ_HMF_PARAMS, logmp_target, redshift
        )
        diff = lg_cuml_density_pred - lj_hmf_cuml_density_target
        mse_diff = np.mean(diff**2)
        assert mse_diff < HMF_FIT_TOL
