# flake8: noqa: E402
""" """
from jax import config

config.update("jax_enable_x64", True)


from collections import namedtuple

from dsps.constants import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from jax import numpy as jnp

from ..phot_utils import get_wave_eff_table
from . import mc_lightcone_halos as mclh
from . import precompute_ssp_phot as psspp
