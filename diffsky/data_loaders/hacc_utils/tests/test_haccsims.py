""" """

import numpy as np
import pytest

from .. import haccsims

try:
    from haccytrees import Simulation as HACCSim

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False

NO_HACC_MSG = "Must have haccytrees installed to run this test"


def test_last_journey_hard_coded():
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


@pytest.mark.skipif(not HAS_HACCYTREES, reason=NO_HACC_MSG)
def test_last_journey_agrees_with_haccytrees():
    sim_name = "LastJourney"
    sim = HACCSim.simulations[sim_name]
    sim2 = haccsims.simulations[sim_name]
    for pname, pval in zip(sim2.cosmo._fields, sim2.cosmo):
        assert np.allclose(pval, getattr(sim.cosmo, pname), rtol=1e-3)

    assert np.allclose(sim.cosmotools_steps, sim2.cosmotools_steps)
    haccytrees_scale_factors = sim.step2a(np.array(sim.cosmotools_steps))
    assert np.allclose(haccytrees_scale_factors, sim2.scale_factors, rtol=1e-4)

    haccytrees_redshifts = sim.step2z(np.array(sim.cosmotools_steps))
    assert np.allclose(haccytrees_redshifts, sim2.redshifts, rtol=1e-4)


@pytest.mark.skipif(not HAS_HACCYTREES, reason=NO_HACC_MSG)
def test_discovery_sims_agree_with_haccytrees():
    for sim_name in ("DiscoveryLCDM", "DiscoveryW0WA"):
        sim = HACCSim.simulations[sim_name]
        sim2 = haccsims.simulations[sim_name]
        for pname, pval in zip(sim2.cosmo._fields, sim2.cosmo):
            assert np.allclose(pval, getattr(sim.cosmo, pname), rtol=1e-3)

        assert np.allclose(sim.cosmotools_steps, sim2.cosmotools_steps)
        haccytrees_scale_factors = sim.step2a(np.array(sim.cosmotools_steps))
        assert np.allclose(haccytrees_scale_factors, sim2.scale_factors, rtol=1e-4)

        haccytrees_redshifts = sim.step2z(np.array(sim.cosmotools_steps))
        assert np.allclose(haccytrees_redshifts, sim2.redshifts, rtol=1e-4)


def test_available_sims():
    assert tuple(haccsims.simulations.keys()) == (
        "LastJourney",
        "DiscoveryLCDM",
        "DiscoveryW0WA",
    )
