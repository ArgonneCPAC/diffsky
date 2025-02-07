"""Utility functions for loading HACC cores"""

import h5py
import numpy as np
from dsps.cosmology import flat_wcdm

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


def scatter_ndarray(array, axis=0, comm=COMM, root=0):
    """Scatter n-dimensional array from root to all ranks

    This function is taken from https://github.com/AlanPearl/diffopt

    """
    ans: np.ndarray = np.array([])
    if comm.rank == root:
        splits = np.array_split(array, comm.size, axis=axis)
        for i in range(comm.size):
            if i == root:
                ans = splits[i]
            else:
                comm.send(splits[i], dest=i)
    else:
        ans = comm.recv(source=root)
    return ans


def load_flat_hdf5(fn, istart=0, iend=None, keys=None):
    """"""

    data = dict()
    with h5py.File(fn, "r") as hdf:

        if keys is None:
            keys = list(hdf.keys())

        for key in keys:
            if iend is None:
                data[key] = hdf[key][istart:]
            else:
                data[key] = hdf[key][istart:iend]

    return data


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

    """
    sim = HACCSim.simulations[sim_name]
    timesteps = np.array(sim.cosmotools_steps)

    a_max = 1 / (1 + z_min)
    a_min = 1 / (1 + z_max)
    a_arr = sim.step2a(timesteps)

    idx_step_min = np.searchsorted(a_arr, a_min) - 1
    idx_step_max = np.searchsorted(a_arr, a_max)
    timestep_min = timesteps[idx_step_min]
    timestep_max = timesteps[idx_step_max]
    return idx_step_min, idx_step_max, timestep_min, timestep_max
