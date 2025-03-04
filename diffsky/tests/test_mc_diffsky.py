"""
"""

import numpy as np
from jax import random as jran

from .. import mc_diffsky as mcd


def test_mc_diffstar_galpop():
    ran_key = jran.key(0)
    hosts_logmh_at_z = np.linspace(10, 15, 200)
    lgmp_min = 11.0
    z_obs = 0.1
    args = (ran_key, z_obs, lgmp_min)
    diffsky_data = mcd.mc_diffstar_galpop(*args, hosts_logmh_at_z=hosts_logmh_at_z)

    for p in diffsky_data["subcat"].mah_params:
        assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(diffsky_data["smh"]))


def test_mc_diffstar_cenpop():
    ran_key = jran.key(0)
    hosts_logmh_at_z = np.linspace(10, 15, 200)
    z_obs = 0.1
    args = (ran_key, z_obs)
    diffsky_data = mcd.mc_diffstar_cenpop(*args, hosts_logmh_at_z=hosts_logmh_at_z)

    for p in diffsky_data["subcat"].mah_params:
        assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(diffsky_data["smh"]))
