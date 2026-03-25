""""""

import numpy as np
from jax import random as jran

from .. import dbpop


def test_frac_disk_dom_kern():
    ngals = 100
    logsm = np.linspace(6, 12, ngals)
    ZZ = np.zeros_like(logsm)

    for logssfr in (-12, -11, -10, -9, -8):
        fdd = dbpop._frac_disk_dom_kern(logsm, logssfr + ZZ)
        assert fdd.shape == logsm.shape
        assert np.all(fdd >= dbpop.FDD_MIN)
        assert np.all(fdd <= dbpop.FDD_MAX)
        assert np.all(np.diff(fdd) <= 0)
        assert np.any(np.diff(fdd) < 0)


def test_get_fbulge_tcrit():
    ran_key = jran.key(0)
    n_gals, n_t = 5_000, 100
    tarr = np.linspace(0.1, 13.8, n_t)

    sfh_key, t_key = jran.split(ran_key, 2)
    sfh_pop = jran.uniform(sfh_key, shape=(n_gals, n_t))

    t_obs_min, t_obs_max = 1.0, 13.0
    t_obs = jran.uniform(t_key, minval=t_obs_min, maxval=t_obs_max, shape=(n_gals,))

    tcrit, logsm_obs, logssfr_obs = dbpop.get_fbulge_tcrit(tarr, sfh_pop, t_obs)
    assert np.all(tcrit < t_obs)
    assert np.all(tcrit > 0)
    assert np.all(np.isfinite(logsm_obs))
    assert np.all(logsm_obs > 0)
    assert np.all(logsm_obs < 20)
    assert np.all(logssfr_obs > -20)
    assert np.all(logssfr_obs < -5)


def test_mc_fbulge_params():
    ran_key = jran.key(0)
    n_gals, n_t = 5_000, 100
    tarr = np.linspace(0.1, 13.8, n_t)
    t_obs_min, t_obs_max = 1.0, 13.0

    sfh_key, t_key = jran.split(ran_key, 2)
    sfh_pop = jran.uniform(sfh_key, shape=(n_gals, n_t))
    t_obs = jran.uniform(t_key, minval=t_obs_min, maxval=t_obs_max, shape=(n_gals,))

    fbulge_params = dbpop.mc_fbulge_params(ran_key, tarr, sfh_pop, t_obs)
