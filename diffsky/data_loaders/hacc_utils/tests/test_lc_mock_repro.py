""" """

import os
from collections import namedtuple

import numpy as np
from diffmah import DEFAULT_MAH_PARAMS
from dsps.constants import T_TABLE_MIN
from dsps.cosmology.flat_wcdm import age_at_z, age_at_z0
from dsps.data_loaders.retrieve_fake_fsps_data import load_fake_ssp_data
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from diffsky.param_utils import diffsky_param_wrapper as dpw

from ....experimental import precompute_ssp_phot as psspp
from ....experimental.disk_bulge_modeling import disk_bulge_kernels as dbk
from ....experimental.disk_bulge_modeling import mc_disk_bulge as mcdb
from ....experimental.lc_phot_kern import get_wave_eff_table
from ....experimental.tests import test_mc_lightcone_halos as tmclh
from ... import io_utils as iou
from .. import lc_mock_repro as lcmp_repro
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

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def test_load_diffsky_param_collection():
    all_params_flat = dpw.unroll_param_collection_into_flat_array(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    all_pnames = dpw.get_flat_param_names()

    Params = namedtuple("Params", all_pnames)
    all_named_params = Params(*all_params_flat)

    drn_mock = ""
    mock_version_name = "unit_testing"
    fn = lcmp_repro.BNPAT_PARAM_COLLECTION.format(mock_version_name)
    iou.write_namedtuple_to_hdf5(all_named_params, fn)

    param_collection = lcmp_repro.load_diffsky_param_collection(
        drn_mock, mock_version_name
    )
    all_params_flat2 = dpw.unroll_param_collection_into_flat_array(*param_collection)

    assert np.allclose(all_params_flat, all_params_flat2, rtol=1e-5)


def _prepare_input_catalogs(n_gals=20):
    lc_data, tcurves = tmclh._get_weighted_lc_data_for_unit_testing(num_halos=n_gals)
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


def test_write_ancillary_data():
    lc_data, diffsky_data, tcurves = _prepare_input_catalogs()
    ssp_data = load_fake_ssp_data()
    mock_version_name = "dummy_mock_version_name"
    sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim("LastJourney")

    drn_mock = os.path.join(_THIS_DRNAME, "tmp_testing")
    os.makedirs(drn_mock, exist_ok=True)
    args = (
        drn_mock,
        mock_version_name,
        sim_info,
        dpw.DEFAULT_PARAM_COLLECTION,
        tcurves,
        ssp_data,
    )
    lcmp_repro.write_ancillary_data(*args)
    t_table = lcmp_repro.load_diffsky_t_table(drn_mock, mock_version_name)
    assert np.all(t_table > 0)
    assert np.all(t_table < 15)
    tcurves2 = lcmp_repro.load_diffsky_tcurves(drn_mock, mock_version_name)
    for name in tcurves2._fields:
        assert np.allclose(getattr(tcurves, name), getattr(tcurves2, name), rtol=0.01)

    ssp_data2 = lcmp_repro.load_diffsky_ssp_data(drn_mock, mock_version_name)
    for name in ssp_data2._fields:
        assert np.allclose(getattr(ssp_data, name), getattr(ssp_data2, name), rtol=0.01)

    param_collection2 = lcmp_repro.load_diffsky_param_collection(
        drn_mock, mock_version_name
    )
    assert np.allclose(
        dpw.DEFAULT_PARAM_COLLECTION.mzr_params, param_collection2.mzr_params
    )


def test_add_dbk_phot_quantities_to_mock():
    diffsky_info = load_lc_cf.get_diffsky_info_from_hacc_sim("LastJourney")

    ssp_data = load_fake_ssp_data()

    lc_data, diffsky_data, tcurves = _prepare_input_catalogs()

    ran_key = jran.key(0)

    z_phot_table = np.linspace(lc_data["z_obs"].min(), lc_data["z_obs"].max(), 3)
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

    _res = lcmp_repro.add_dbk_phot_quantities_to_mock(*args)
    phot_info, lc_data, diffsky_data = _res

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
        t_table, phot_info["sfh_table"]
    )

    for pname in disk_bulge_history.fbulge_params._fields:
        assert np.allclose(
            getattr(disk_bulge_history.fbulge_params, pname),
            getattr(fbulge_params, pname),
            rtol=0.01,
        )
