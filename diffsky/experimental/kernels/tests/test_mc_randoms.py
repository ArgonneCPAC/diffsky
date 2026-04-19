""""""

import numpy as np
from diffmah import DEFAULT_MAH_PARAMS
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from jax import random as jran

from ....param_utils import diffsky_param_wrapper as dpw
from ...disk_bulge_modeling import disk_knots
from .. import mc_randoms


def test_get_mc_phot_randoms():
    ran_key = jran.key(0)
    n_gals = 2_000
    ZZ = np.zeros(n_gals)
    mah_params = DEFAULT_MAH_PARAMS._make([ZZ + x for x in DEFAULT_MAH_PARAMS])
    phot_randoms, sfh_params = mc_randoms.get_mc_phot_randoms(
        ran_key,
        dpw.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
        mah_params,
        DEFAULT_COSMOLOGY,
    )
    assert np.all(np.unique(phot_randoms.mc_is_q) == np.array((0, 1)))
    for key in ("uran_av", "uran_delta", "uran_funo", "uran_pburst"):
        randoms = getattr(phot_randoms, key)
        assert np.all(randoms > 0)
        assert np.all(randoms < 1)
        assert np.allclose(randoms.mean(), 0.5, atol=0.01)


def test_get_mc_dbk_randoms():
    ran_key = jran.key(0)
    n_gals = 2_000
    dbk_randoms = mc_randoms.get_mc_dbk_randoms(ran_key, n_gals)
    assert np.all(dbk_randoms.fknot < disk_knots.FKNOT_MAX)
    assert np.all(dbk_randoms.fknot > 0)

    assert np.all(dbk_randoms.uran_fbulge > 0)
    assert np.all(dbk_randoms.uran_fbulge < 1)
    assert np.allclose(dbk_randoms.uran_fbulge.mean(), 0.5, atol=0.01)


def test_get_merging_randoms():
    ran_key = jran.key(0)
    n_gals = 2_000
    merging_randoms_randoms = mc_randoms.get_merging_randoms(ran_key, n_gals)

    assert np.all(merging_randoms_randoms.uran_pmerge > 0)
    assert np.all(merging_randoms_randoms.uran_pmerge < 1)
    assert np.allclose(merging_randoms_randoms.uran_pmerge.mean(), 0.5, atol=0.01)
