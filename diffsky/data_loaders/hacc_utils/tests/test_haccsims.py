""" """

import numpy as np

from .. import haccsims


def test_last_journey():
    sim = haccsims.simulations["LastJourney"]
    assert sim.rl == 3400
    assert np.allclose(sim.particle_mass, 10**9.434, rtol=0.01)
    assert sim.np == 10752
