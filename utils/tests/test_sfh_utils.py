""" """

import numpy as np
from jax import random as jran

from .. import sfh_utils


def test_get_logsm_logssfr_at_t_obs():
    ran_key = jran.key(0)
    sfh_key, t_obs_key = jran.split(ran_key, 2)

    n_gals = 500
    n_t = 150
    t_table = np.linspace(0.01, 13.8, n_t)
    t_obs = jran.uniform(t_obs_key, minval=0.1, maxval=13.8, shape=(n_gals,))
    sfh = jran.uniform(sfh_key, minval=0, maxval=100, shape=(n_gals, n_t))

    logsm_obs, logssfr_obs = sfh_utils.get_logsm_logssfr_at_t_obs(t_obs, t_table, sfh)

    assert np.all(np.isfinite(logsm_obs))
    assert np.all(logsm_obs > 5)
    assert np.all(logsm_obs < 13)

    assert np.all(np.isfinite(logssfr_obs))
    assert np.all(logssfr_obs > -15)
    assert np.all(logssfr_obs < -5)
