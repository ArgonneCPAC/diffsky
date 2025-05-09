""" """

import numpy as np
import pytest
from jax import random as jran

from .. import hacc_core_utils as hcu

try:
    from haccytrees import Simulation as HACCSim

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False
NO_HACC_MSG = "Must have haccytrees installed to run this test"


@pytest.mark.skipif(not HAS_HACCYTREES, reason=NO_HACC_MSG)
def test_get_diffstar_cosmo_quantities():
    sim_name = "LastJourney"
    fb, lgt0 = hcu.get_diffstar_cosmo_quantities(sim_name)
    assert 0.1 < fb < 0.2
    assert 1.1 < lgt0 < 1.2


@pytest.mark.skipif(not HAS_HACCYTREES, reason=NO_HACC_MSG)
def test_get_timestep_range_from_z_range():
    sim_name = "LastJourney"
    sim = HACCSim.simulations[sim_name]
    timesteps = np.array(sim.cosmotools_steps)
    z_arr = sim.step2z(timesteps)

    Z_MIN, Z_MAX = z_arr.min() + 0.02, z_arr.max() - 0.02

    ran_key = jran.key(0)
    n_tests = 1000
    for __ in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        z_min, z_max = jran.uniform(test_key, minval=Z_MIN, maxval=Z_MAX, shape=(2,))
        z_min, z_max = np.sort((z_min, z_max))

        _res = hcu.get_timestep_range_from_z_range(sim_name, z_min, z_max)
        idx_step_min, idx_step_max, timestep_min, timestep_max = _res

        # Enforce z_min is contained in the step range
        snap_z_lo = z_arr[idx_step_max]
        assert z_min >= snap_z_lo

        # Enforce z_max is contained in the step range
        snap_z_hi = z_arr[idx_step_min]
        assert z_max <= snap_z_hi

        # Enforce the next timestep would have overshot z_min
        if idx_step_max < z_arr.size - 1:
            assert z_min < z_arr[idx_step_max - 1]

        # Enforce the next timestep would have undershot z_max
        if idx_step_max > 0:
            assert z_max > z_arr[idx_step_min + 1]


@pytest.mark.skipif(not HAS_HACCYTREES, reason=NO_HACC_MSG)
def test_get_timestep_range_from_z_range_edge_cases():
    sim_name = "LastJourney"
    sim = HACCSim.simulations[sim_name]
    timesteps = np.array(sim.cosmotools_steps)
    z_arr = sim.step2z(timesteps)

    # z_min = 0.0
    z_min, z_max = 0.0, 3.0
    _res = hcu.get_timestep_range_from_z_range(sim_name, z_min, z_max)
    idx_step_min, idx_step_max, timestep_min, timestep_max = _res
    assert timestep_max == timesteps[-1]

    # z_max > z_arr.max()
    z_min, z_max = 0.1, z_arr.max() + 10.0
    _res = hcu.get_timestep_range_from_z_range(sim_name, z_min, z_max)
    idx_step_min, idx_step_max, timestep_min, timestep_max = _res
    assert timestep_min == timesteps[0]
