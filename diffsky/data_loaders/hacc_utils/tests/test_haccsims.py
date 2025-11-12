""" """

import numpy as np

from .. import haccsims


def test_last_journey():
    sim = haccsims.simulations["LastJourney"]
    assert sim.rl == 3400
    assert np.allclose(sim.particle_mass, 10**9.434, rtol=0.01)
    assert sim.np == 10752

    assert np.allclose(sim.cosmo.h, 0.6766, rtol=1e-3)
    assert np.allclose(sim.cosmo.Omega_m, 0.30964468, rtol=1e-3)
    assert np.allclose(sim.cosmo.w0, -1.0, rtol=1e-3)
    assert np.allclose(sim.cosmo.wa, 0.0, rtol=1e-3)
    assert np.allclose(sim.cosmo.Omega_b, 0.04897468, rtol=1e-3)
    assert np.allclose(sim.cosmo.s8, 0.8102, rtol=1e-3)
    assert np.allclose(sim.cosmo.ns, 0.9665, rtol=1e-3)

    assert np.all(np.diff(sim.cosmotools_steps) > 0)
    assert np.all(sim.cosmotools_steps > 0)
    assert np.allclose(sim.cosmotools_steps[-2:], np.array((487, 499)))

    assert sim.scale_factors.shape == (101,)
    assert np.allclose(1.0 / (1.0 + sim.redshifts), sim.scale_factors, rtol=1e-4)

    assert np.allclose(sim.redshifts[-1], 0.0, atol=1e-4)


def test_available_sims():
    assert tuple(haccsims.simulations.keys()) == ("LastJourney",)
