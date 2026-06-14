""""""

import numpy as np
from jax import random as jran

from .. import dbk_masses as dbkm


def test_get_disk_bulge_knot_masses():
    ran_key = jran.key(0)
    logsm_key, bt_key, fknot_key = jran.split(ran_key, 3)
    n = 2_000
    logsm_obs = jran.uniform(logsm_key, minval=5, maxval=12, shape=(n,))
    bt = jran.uniform(bt_key, minval=0, maxval=1, shape=(n,))
    fknot = jran.uniform(fknot_key, minval=0, maxval=0.5, shape=(n,))
    mstar_tot = 10**logsm_obs
    mb_correct = bt * mstar_tot
    mdtot_correct = mstar_tot - mb_correct
    mk_correct = fknot * mdtot_correct
    md_correct = mdtot_correct - mk_correct

    mock = dict(logsm_obs=logsm_obs, bulge_to_total=bt, fknot=fknot)
    dbk_mass = dbkm.get_disk_bulge_knot_masses(mock)

    assert set(dbk_mass.keys()) == set(dbkm.DBKMassKeys)
    assert np.allclose(dbk_mass["mstar_bulge"], mb_correct, rtol=1e-7)
    assert np.allclose(dbk_mass["mstar_diffuse_disk"], md_correct, rtol=1e-7)
    assert np.allclose(dbk_mass["mstar_knots"], mk_correct, rtol=1e-7)
