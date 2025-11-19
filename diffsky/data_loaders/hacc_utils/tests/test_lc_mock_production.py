""" """

import os
from copy import deepcopy

import numpy as np
import pytest
from diffmah import DEFAULT_MAH_PARAMS
from dsps.constants import T_TABLE_MIN
from dsps.cosmology.flat_wcdm import age_at_z, age_at_z0
from dsps.data_loaders import load_ssp_templates
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from diffsky.param_utils import diffsky_param_wrapper as dpw

from ....experimental import precompute_ssp_phot as psspp
from ....experimental.disk_bulge_modeling import disk_bulge_kernels as dbk
from ....experimental.disk_bulge_modeling import mc_disk_bulge as mcdb
from ....experimental.lc_phot_kern import get_wave_eff_table
from ....experimental.tests import test_lc_phot_kern as tlcphk
from .. import lc_mock_production as lcmp
from .. import load_lc_cf

vmap_interp = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))

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
def test_add_sfh_quantities_to_mock_last_journey():
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


def test_add_sfh_quantities_to_mock():
    ran_key = jran.key(0)
    diffsky_info = load_lc_cf.get_diffsky_info_from_hacc_sim("LastJourney")
    lc_data, diffsky_data, tcurves = _prepare_input_catalogs()
    lc_data, diffsky_data = lcmp.add_sfh_quantities_to_mock(
        diffsky_info, lc_data, diffsky_data, ran_key
    )

    sfh_table_recomputed = np.where(
        diffsky_data["mc_is_q"].reshape((-1, 1)),
        diffsky_data["sfh_table_q"],
        diffsky_data["sfh_table_ms"],
    )
    assert np.allclose(diffsky_data["sfh_table"], sfh_table_recomputed, rtol=0.01)


def test_add_dbk_sed_quantities_to_mock():
    diffsky_info = load_lc_cf.get_diffsky_info_from_hacc_sim("LastJourney")

    ssp_data = load_ssp_templates()

    lc_data, diffsky_data, tcurves = _prepare_input_catalogs()
    diffsky_data_orig = deepcopy(diffsky_data)

    __, sfh_key = jran.split(jran.key(0), 2)
    lc_data, diffsky_data = lcmp.add_sfh_quantities_to_mock(
        diffsky_info, deepcopy(lc_data), deepcopy(diffsky_data), sfh_key
    )
    diffsky_data_with_sfh = deepcopy(diffsky_data)

    z_phot_table = np.linspace(lc_data["z_obs"].min(), lc_data["z_obs"].max(), 15)
    t0 = age_at_z0(*diffsky_info.cosmo_params)
    t_table = np.linspace(T_TABLE_MIN, t0, 100)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, diffsky_info.cosmo_params
    )
    wave_eff_table = get_wave_eff_table(z_phot_table, tcurves)

    ran_key = jran.key(0)
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

    assert np.allclose(
        diffsky_data_with_sfh["sfh_table"], diffsky_data["sfh_table"], rtol=0.01
    )

    fbulge_params = dbk.DEFAULT_FBULGE_PARAMS._make(
        (phot_info["fbulge_tcrit"], phot_info["fbulge_early"], phot_info["fbulge_late"])
    )

    t_obs = age_at_z(lc_data["redshift_true"], *diffsky_info.cosmo_params)

    _res = dbk._bulge_sfh_vmap(t_table, phot_info["sfh_table"], fbulge_params)
    bth = _res[-1]
    bulge_to_total_recomputed = vmap_interp(t_obs, t_table, bth)
    bulge_to_total_recomputed2 = vmap_interp(
        t_obs, t_table, phot_info["bulge_to_total_history"]
    )
    assert np.allclose(bulge_to_total_recomputed, bulge_to_total_recomputed2, rtol=0.01)

    disk_bulge_history = mcdb.decompose_sfh_into_disk_bulge_sfh(
        t_table, diffsky_data["sfh_table"]
    )

    for pname in disk_bulge_history.fbulge_params._fields:
        assert np.allclose(
            getattr(disk_bulge_history.fbulge_params, pname),
            getattr(fbulge_params, pname),
            rtol=0.01,
        )
