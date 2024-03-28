"""
"""

import numpy as np
from jax import random as jran

from ..mc_tinfall import mc_time_since_infall


def test_mc_infall_time_has_non_stochastic_behavior():
    ran_key = jran.PRNGKey(0)

    rng0 = np.random.RandomState(0)
    rng1 = np.random.RandomState(1)

    nsubs = 250_000
    ran_key, mu_key, tobs_key = jran.split(ran_key, 3)
    lgmu = jran.uniform(mu_key, minval=-5, maxval=0, shape=(nsubs,))
    t_obs = 10.0
    time_since_infall0 = mc_time_since_infall(lgmu, t_obs, random_state=rng0)
    time_since_infall1 = mc_time_since_infall(lgmu, t_obs, random_state=rng1)
    time_since_infall2 = mc_time_since_infall(lgmu, t_obs, random_state=rng0)
    assert not np.allclose(time_since_infall0, time_since_infall1)
    assert np.allclose(time_since_infall0, time_since_infall2)


def test_mc_infall_time_behaves_as_expected():
    ran_key = jran.PRNGKey(0)

    for i in range(1000):
        rng_np = np.random.RandomState(i)

        # Scalar inputs should return scalar output
        lgmu, t_obs = -1.0, 10.0
        time_since_infall = mc_time_since_infall(lgmu, t_obs, random_state=rng_np)
        assert time_since_infall.shape == ()

        # Generate subhalo population at single t_obs
        nsubs = 250_000
        ran_key, mu_key, tobs_key = jran.split(ran_key, 3)
        lgmu = jran.uniform(mu_key, minval=-5, maxval=0, shape=(nsubs,))
        time_since_infall = mc_time_since_infall(lgmu, t_obs, random_state=rng_np)

        # infall times should respect 0 < time_since_infall < t_obs
        assert time_since_infall.shape == (nsubs,)
        assert np.all(np.isfinite(time_since_infall))
        assert np.all(time_since_infall < t_obs)
        assert np.all(time_since_infall > 0)

        # Larger subhalos should on average have more recent infall times
        lgmuarr = np.linspace(-4, -0.5, 5)
        mean_tinf = [time_since_infall[np.abs(lgmu - x) < 0.25].mean() for x in lgmuarr]
        assert np.all(np.diff(mean_tinf) < 0), "iseed = {0}".format(i)

        # Generate subhalo population on a lightcone
        # infall times should respect 0 < time_since_infall < t_obs
        t_obs = jran.uniform(tobs_key, minval=1, maxval=14, shape=(nsubs,))
        time_since_infall = mc_time_since_infall(lgmu, t_obs, random_state=rng_np)
        assert time_since_infall.shape == (nsubs,)
        assert np.all(np.isfinite(time_since_infall))
        assert np.all(time_since_infall < t_obs)
        assert np.all(time_since_infall > 0)
