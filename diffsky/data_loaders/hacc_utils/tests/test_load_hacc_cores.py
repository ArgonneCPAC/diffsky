""" """

import os

import numpy as np
import pytest
from dsps.cosmology import flat_wcdm
from jax import random as jran

from ....mass_functions.mc_diffmah_tpeak import mc_subhalos
from .. import load_hacc_cores as lhc

NO_HACC_MSG = "Must have haccytrees installed to run this test"
POBOY_MSG = "This test only runs on poboy machine"

DRN_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"
DRN_LJ_DMAH_POBOY = "/Users/aphearin/work/DATA/LastJourney/diffmah_fits"

BNPAT_CORE_DATA = "m000p.coreforest.{}.hdf5"

DRN_DISCOVERY_POBOY = "/Users/aphearin/work/DATA/DESI_W0WA"

try:
    assert os.path.isdir(DRN_LJ_POBOY)
    assert os.path.isdir(DRN_DISCOVERY_POBOY)
    assert lhc.HAS_HACCYTREES
    CAN_RUN_HACC_DATA_TESTS = True
except AssertionError:
    CAN_RUN_HACC_DATA_TESTS = False


def test_concatenate_diffsky_subcats():
    n_cats = 3
    ran_key = jran.key(0)
    z_obs = 0.05
    lgmp_min = 11.5
    volume_com = 25**3
    subcats = []
    for i in range(n_cats):
        ran_key, cat_key = jran.split(ran_key, 2)
        catdata = mc_subhalos(cat_key, z_obs, lgmp_min, volume_com)._asdict()
        catdata.pop("halo_ids")
        catdata.pop("host_mah_params")
        catdata["fake_mah"] = np.ones(len(catdata["logmp0"])).astype(int)
        cat = lhc.SubhaloCatalog._make(list(catdata.values()))
        subcats.append(cat)

    subcat = lhc.concatenate_diffsky_subcats(subcats)

    # Enforce that the host index was correctly computed
    host_logmp0_correct = np.concatenate(
        [cat.logmp0[cat.ult_host_indx] for cat in subcats]
    )
    assert np.allclose(subcat.logmp0[subcat.ult_host_indx], host_logmp0_correct)


@pytest.mark.skipif(not CAN_RUN_HACC_DATA_TESTS, reason=POBOY_MSG)
def test_load_last_journey_data():
    sim_name = "LastJourney"
    subvol = 0
    chunknum = 49
    nchunks = 50
    iz_obs = 100
    ran_key = jran.key(0)
    drn_cores = DRN_LJ_POBOY
    drn_diffmah = DRN_LJ_DMAH_POBOY
    diffsky_data = lhc.load_diffsky_data(
        sim_name, subvol, chunknum, nchunks, iz_obs, ran_key, drn_cores, drn_diffmah
    )
    for x in diffsky_data["subcat"].mah_params:
        assert np.all(np.isfinite(x))

    for x in diffsky_data["subcat"][1:]:
        assert np.all(np.isfinite(x))

    n_diffmah_fits = diffsky_data["subcat"].mah_params.logm0.size
    n_forest = diffsky_data["subcat"].logmp0.size
    assert n_forest == n_diffmah_fits, "mismatch between forest and diffmah fits"


@pytest.mark.skipif(not CAN_RUN_HACC_DATA_TESTS, reason=POBOY_MSG)
def test_load_coreforest_and_metadata_discovery_sims():

    chunknum = 0
    nchunks = 100
    bname_89 = "m000p.coreforest.89.hdf5"
    for sim_nickname in ("LCDM", "W0WA"):
        sim_name = "Discovery" + sim_nickname
        drn = os.path.join(DRN_DISCOVERY_POBOY, sim_nickname)
        fn_cores = os.path.join(drn, bname_89)
        _res = lhc.load_coreforest_and_metadata(fn_cores, sim_name, chunknum, nchunks)
        sim, cosmo_dsps, forest_matrices, zarr, tarr, lgt0 = _res
        assert 0.9 < lgt0 < 1.2
        assert np.allclose(tarr[-1], 10**lgt0, rtol=0.01)
        assert "central" in forest_matrices.keys()
        tarr2 = flat_wcdm.age_at_z(zarr, *cosmo_dsps)
        assert np.allclose(tarr, tarr2, rtol=1e-3)

        assert np.allclose(sim.cosmo.Omega_m, cosmo_dsps.Om0, rtol=1e-4)
        assert np.allclose(sim.cosmo.w0, cosmo_dsps.w0, rtol=1e-4)
        assert np.allclose(sim.cosmo.wa, cosmo_dsps.wa, rtol=1e-4)
        assert np.allclose(sim.cosmo.h, cosmo_dsps.h, rtol=1e-4)


@pytest.mark.skipif(not CAN_RUN_HACC_DATA_TESTS, reason=POBOY_MSG)
def test_load_diffsky_data():
    ran_key = jran.key(0)

    for sim_nickname in ("LCDM", "W0WA"):
        sim_name = "Discovery" + sim_nickname
        subvol = 89
        chunknum = 0
        nchunks = 50
        iz_obs = 80

        drn_cores = os.path.join(DRN_DISCOVERY_POBOY, sim_nickname)
        drn_diffmah = drn_cores
        args = (
            sim_name,
            subvol,
            chunknum,
            nchunks,
            iz_obs,
            ran_key,
            drn_cores,
            drn_diffmah,
        )

        diffsky_data = lhc.load_diffsky_data(*args)
        for key in lhc.DIFFSKY_DATA_DICT_KEYS:
            assert key in diffsky_data.keys()
