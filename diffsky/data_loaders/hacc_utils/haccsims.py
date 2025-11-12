"""Module storing HACC simulation info"""

import os
from collections import namedtuple

import numpy as np

CosmoParams = namedtuple("CosmoParams", ("Omega_m", "w0", "wa", "h", "Omega_b"))
SimSpecs = namedtuple("SimSpecs", ("rl", "np", "particle_mass"))

_THIS_DRN = os.path.dirname(os.path.abspath(__file__))

SIM_INFO = dict(LastJourney=SimSpecs(3400, 10752, 2717395894.4894614))


class HACCSim(object):

    def __init__(self, sim_name):
        self._cosmotools_steps = None
        self._cosmo = None
        self._drn_simdata = os.path.join(_THIS_DRN, "data", sim_name)

        sim_info = SIM_INFO[sim_name]
        for key, val in zip(sim_info._fields, sim_info):
            setattr(self, key, val)

    @classmethod
    def cosmotools_steps(self):
        if self._cosmotools_steps is None:
            fn = os.path.join(self._drn_simdata, "cosmotools_steps.txt")
            self._cosmotools_steps = np.loadtxt(fn)
        return self._cosmotools_steps

    @classmethod
    def cosmo(self):
        if self._cosmo is None:
            fn = os.path.join(self._drn_simdata, "cosmo.txt")
            self._cosmo = CosmoParams(*np.loadtxt(fn))
        return self._cosmo

    def step2z(self):
        raise NotImplementedError()

    def step2a(self):
        raise NotImplementedError()


simulations = dict()
for sim_name, sim_info in SIM_INFO.items():
    simulations[sim_name] = HACCSim(sim_name)
