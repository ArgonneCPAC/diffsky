"""
"""
import numpy as np
from jax import random as jran

from ..mc_hosts import LGMH_MAX, mc_host_halos_singlez


def test_mc_host_halo_logmp_behaves_as_expected():
    ran_key = jran.PRNGKey(0)

    lgmp_min, redshift, volume_com = 11.0, 0.1, 100**3
    lgmp_halopop = mc_host_halos_singlez(ran_key, lgmp_min, redshift, volume_com)
    assert lgmp_halopop.size > 0
    assert np.all(lgmp_halopop > lgmp_min)
    assert np.all(lgmp_halopop < LGMH_MAX)
