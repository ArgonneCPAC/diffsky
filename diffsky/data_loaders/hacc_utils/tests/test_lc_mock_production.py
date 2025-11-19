""" """

import os

import numpy as np
import pytest
from diffmah import DEFAULT_MAH_PARAMS
from dsps.data_loaders import load_ssp_templates
from jax import random as jran

from diffsky.param_utils import diffsky_param_wrapper as dpw

from ....experimental import precompute_ssp_phot as psspp
from ....experimental.lc_phot_kern import get_wave_eff_table
from ....experimental.tests import test_lc_phot_kern as tlcphk
from .. import lc_mock_production as lcmp
from .. import load_lc_cf

DRN_CF_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"
DRN_LC_CF_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney/lc-cf-diffsky"

HAS_HACCY_TREES = load_lc_cf.HAS_HACCYTREES

try:
    assert os.path.isdir(DRN_CF_LJ_POBOY)
    CAN_RUN_LJ_DATA_TESTS = True
except AssertionError:
    CAN_RUN_LJ_DATA_TESTS = False
CAN_RUN_LJ_DATA_TESTS = CAN_RUN_LJ_DATA_TESTS & HAS_HACCY_TREES
POBOY_MSG = "This test only runs on poboy machine with haccytrees installed"


@pytest.mark.skipif(not CAN_RUN_LJ_DATA_TESTS, reason=POBOY_MSG)
def test_add_sfh_quantities_to_mock():
    ran_key = jran.key(0)
    sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim("LastJourney")

    bn_list = ["lc_cores-213.0.diffsky_data.hdf5"]
    fn_list = [os.path.join(DRN_LC_CF_LJ_POBOY, bn) for bn in bn_list]
    lc_data, diffsky_data = load_lc_cf.collect_lc_diffsky_data(fn_list)

    args = (sim_info, lc_data, diffsky_data, ran_key)
    lc_data, diffsky_data = lcmp.add_sfh_quantities_to_mock(*args)


def _prepare_input_catalogs(n_gals=500):
    lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing(num_halos=n_gals)
    lc_data = lc_data._asdict()
    lc_data["redshift_true"] = lc_data["z_obs"]

    ZZ = np.zeros(n_gals).astype(int)

    diffsky_data = dict()
    diffsky_data["n_points_per_fit"] = ZZ + 100
    diffsky_data["loss"] = 1e-9

    for key in DEFAULT_MAH_PARAMS._fields:
        diffsky_data[key] = getattr(lc_data["mah_params"], key)

    lc_data["central"] = ZZ + 1

    return lc_data, diffsky_data, tcurves


def test_add_dbk_sed_quantities_to_mock():
    ran_key = jran.key(0)
    diffsky_info = load_lc_cf.get_diffsky_info_from_hacc_sim("LastJourney")

    ssp_data = load_ssp_templates()

    lc_data, diffsky_data, tcurves = _prepare_input_catalogs()

    z_phot_table = np.linspace(lc_data["z_obs"].min(), lc_data["z_obs"].max(), 15)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, diffsky_info.cosmo_params
    )
    wave_eff_table = get_wave_eff_table(z_phot_table, tcurves)

    args = (
        diffsky_info,
        lc_data,
        diffsky_data,
        ssp_data,
        dpw.DEFAULT_PARAM_COLLECTION,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        ran_key,
    )
    _res = lcmp.add_dbk_sed_quantities_to_mock(*args)
    phot_info, lc_data, diffsky_data = _res
