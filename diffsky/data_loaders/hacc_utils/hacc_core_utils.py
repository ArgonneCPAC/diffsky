"""Utility functions for loading HACC cores"""

import numpy as np
from dsps.cosmology import flat_wcdm

from ..mpi_utils import scatter_ndarray

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:
    MPI = COMM = None

try:
    from haccytrees import Simulation as HACCSim

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False


def scatter_subcat(subcat, comm):
    mah_params = scatter_mah_params(subcat.mah_params, comm)
    seq = []
    for arr in subcat[1:]:
        seq.append(scatter_ndarray(arr, axis=0, comm=comm, root=0))
    data = [mah_params, *seq]
    subcat = subcat._make(data)
    return subcat


def scatter_mah_params(mah_params, comm):
    seq = []
    for key, arr in zip(mah_params._fields, mah_params):
        seq.append(scatter_ndarray(arr, axis=0, comm=comm, root=0))
    mah_params = mah_params._make(seq)
    return mah_params


def get_diffstar_cosmo_quantities(sim_name):
    sim = HACCSim.simulations[sim_name]
    fb = sim.cosmo.Omega_b / sim.cosmo.Omega_m

    cosmo_dsps = flat_wcdm.CosmoParams(
        *(sim.cosmo.Omega_m, sim.cosmo.w0, sim.cosmo.wa, sim.cosmo.h)
    )
    t0 = flat_wcdm.age_at_z0(*cosmo_dsps)
    lgt0 = np.log10(t0)

    return fb, lgt0


def get_timestep_range_from_z_range(sim_name, z_min, z_max):
    """Find the HACC timesteps that contains the redshift range [z_min, z_max]

    Parameters
    ----------
    sim_name : HACCSim name

    z_min, z_max : floats

    Returns
    -------
    idx_step_min, idx_step_max : ints
        Indices of the timestep array that span the input z-range

    timestep_min, timestep_max : ints
        Timesteps that span the input z-range

    Notes
    -----
    sim = HACCSim.simulations[sim_name]
    timesteps = np.array(sim.cosmotools_steps)
    timestep_min = timesteps[idx_step_min]
    timestep_max = timesteps[idx_step_max]

    """
    sim = HACCSim.simulations[sim_name]
    timesteps = np.array(sim.cosmotools_steps)
    z_arr = sim.step2z(timesteps)

    a_max = 1 / (1 + z_min)
    a_min = 1 / (1 + z_max)
    a_arr = sim.step2a(timesteps)

    if a_min < a_arr[0]:
        idx_step_min = 0
        print(f"Input z_max={z_max:.4f}>{z_arr.max():.4f} = largest lightcone redshift")
        print(f"Using z_max={z_arr.max():.4f}")
    else:
        idx_step_min = np.searchsorted(a_arr, a_min) - 1

    if a_max > a_arr[-1]:
        idx_step_max = len(timesteps) - 1
        print(f"Input z_min={z_min} is less than smallest lightcone redshift")
        print(f"Using z_min={z_arr.min():.4e}")
    else:
        idx_step_max = np.searchsorted(a_arr, a_max)

    timestep_min = timesteps[idx_step_min]
    timestep_max = timesteps[idx_step_max]

    return idx_step_min, idx_step_max, timestep_min, timestep_max
