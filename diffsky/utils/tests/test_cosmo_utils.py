""""""

import numpy as np
from jax import random as jran

from .. import cosmo_utils


def test_get_redshift_obs_from_redshift_true():
    ran_key = jran.key(0)
    ngals = 100_000
    z_min, z_max = 0.0001, 5.0

    z_key, v_key = jran.split(ran_key, 2)
    z_true = jran.uniform(z_key, minval=z_min, maxval=z_max, shape=(ngals,))

    vpec_std = 1_000.0
    vpec_kms = jran.normal(v_key, shape=(ngals,)) * vpec_std
    redshift_obs = cosmo_utils.get_redshift_obs_from_redshift_true(z_true, vpec_kms)
    assert np.all(np.isfinite(redshift_obs))
    assert np.max(np.abs(redshift_obs - z_true)) < 0.2
    assert np.abs(np.mean(redshift_obs - z_true)) < 1e-4
