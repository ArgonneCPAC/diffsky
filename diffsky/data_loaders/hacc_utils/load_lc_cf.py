""""""

from collections import namedtuple

import numpy as np
from dsps.cosmology import flat_wcdm

try:
    from haccytrees import Simulation as HACCSim

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False

SIM_INFO_KEYS = ("sim", "cosmo_params", "z_sim", "t_sim", "lgt0", "fb")
DiffskySimInfo = namedtuple("DiffskySimInfo", SIM_INFO_KEYS)


def get_diffsky_info_from_hacc_sim(sim_name):
    sim = HACCSim.simulations[sim_name]

    cosmo_params = flat_wcdm.CosmoParams(
        *(sim.cosmo.Omega_m, sim.cosmo.w0, sim.cosmo.wa, sim.cosmo.h)
    )
    z_sim = np.array(sim.step2z(np.array(sim.cosmotools_steps)))
    t_sim = flat_wcdm.age_at_z(z_sim, *cosmo_params)

    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = np.log10(t0)
    fb = sim.cosmo.Omega_b / sim.cosmo.Omega_m

    diffsky_info = DiffskySimInfo(sim, cosmo_params, z_sim, t_sim, lgt0, fb)

    return diffsky_info
