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
