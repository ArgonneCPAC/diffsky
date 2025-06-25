""""""

import os
from glob import glob

import numpy as np

from ...hmf_model import predict_cuml_hmf
from .. import smdpl_hmf
from ..smdpl_hmf_fitting_helpers import get_z_from_bn

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DRN_TESTING_DATA = os.path.join(_THIS_DRNAME, "testing_data")


def test_hmf_model_accuracy():
    fn_list = sorted(
        glob(os.path.join(DRN_TESTING_DATA, "*subhalos.lgcuml_density*.npy"))
    )
    bn_list = [os.path.basename(fn) for fn in fn_list]
    fn_list_lgmp = [s.replace("lgcuml_density", "logmp_bins") for s in fn_list]

    z_list = np.array([get_z_from_bn(bname) for bname in bn_list])

    loss_data_collector = []
    for iz, (fn, fn_lgmp) in enumerate(zip(fn_list, fn_list_lgmp)):
        lgcuml_density = np.load(fn)
        lgmp_bins = np.load(fn_lgmp)
        loss_data_iz = (z_list[iz], lgmp_bins, lgcuml_density)
        loss_data_collector.append(loss_data_iz)
