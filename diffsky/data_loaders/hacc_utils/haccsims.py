"""Module storing HACC simulation info"""

import os
from collections import namedtuple

import numpy as np

CosmoParams = namedtuple("CosmoParams", ("Omega_m", "w0", "wa", "h", "Omega_b"))

_THIS_DRN = os.path.dirname(os.path.abspath(__file__))

SIM_NAMES = ("LastJourney",)


class HACCSim(object):

    def __init__(self, sim_name):
        self._cosmotools_steps = None
        self._cosmo = None
        self._drn_simdata = os.path.join(_THIS_DRN, "data", sim_name)

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
for sim_name in SIM_NAMES:
    simulations[sim_name] = HACCSim(sim_name)
