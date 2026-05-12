"""Module storing HACC simulation info"""

import os
from collections import namedtuple

import numpy as np

CosmoParams = namedtuple(
    "CosmoParams", ("Omega_m", "w0", "wa", "h", "Omega_b", "s8", "ns")
)
SimSpecs = namedtuple("SimSpecs", ("rl", "np", "particle_mass"))

_THIS_DRN = os.path.dirname(os.path.abspath(__file__))

SIM_INFO = dict(
    LastJourney=SimSpecs(3400, 10752, 2717395894.4894614),
    DiscoveryLCDM=SimSpecs(1019.55, 6720, 297466442.6322133),
    DiscoveryW0WA=SimSpecs(969.9, 6720, 286943385.1781562),
)


class HACCSim(object):

    def __init__(self, sim_name):
        self._cosmotools_steps = None
        self._cosmo = None
        self._scale_factors = None

        self._drn_simdata = os.path.join(_THIS_DRN, "data", sim_name)

        sim_info = SIM_INFO[sim_name]
        for key, val in zip(sim_info._fields, sim_info):
            setattr(self, key, val)

    @property
    def cosmotools_steps(self):
        if self._cosmotools_steps is None:
            fn = os.path.join(self._drn_simdata, "cosmotools_steps.txt")
            self._cosmotools_steps = np.loadtxt(fn).astype(int)
        return self._cosmotools_steps

    @property
    def cosmo(self):
        if self._cosmo is None:
            fn = os.path.join(self._drn_simdata, "cosmo.txt")
            self._cosmo = CosmoParams(*np.loadtxt(fn))
        return self._cosmo

    @property
    def scale_factors(self):
        if self._scale_factors is None:
            fn = os.path.join(self._drn_simdata, "cosmotools_steps_a.txt")
            self._scale_factors = np.loadtxt(fn)
        return self._scale_factors

    @property
    def redshifts(self):
        return 1.0 / self.scale_factors - 1.0


simulations = dict()
for sim_name, sim_info in SIM_INFO.items():
    simulations[sim_name] = HACCSim(sim_name)
