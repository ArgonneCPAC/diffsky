"""
"""

import numpy as np
from jax import random as jran

from .. import mc_diffsky as mcd


def test_mc_diffstar_galhalo_pop():
    ran_key = jran.key(0)
    hosts_logmh_at_z = np.linspace(10, 15, 200)
    lgmp_min = 11.0
    z_obs = 0.1
    args = (ran_key, lgmp_min, z_obs)
    diffsky_data = mcd.mc_diffstar_galhalo_pop(*args, hosts_logmh_at_z=hosts_logmh_at_z)
