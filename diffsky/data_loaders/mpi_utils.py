"""Utility functions for use with mpi4py"""

import numpy as np

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:
    MPI = COMM = None


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


def get_mpi_rank_info(comm):
    rank = comm.Get_rank()
    node_name = MPI.Get_processor_name()
    nodelist = comm.allgather(node_name)
    unique_nodelist = sorted(list(set(nodelist)))
    node_number = unique_nodelist.index(node_name)
    intra_node_id = len([i for i in nodelist[:rank] if i == node_name])

    rankinfo = (rank, intra_node_id, node_number)
    infolist = comm.allgather(rankinfo)
    sorted_infolist = sorted(infolist, key=lambda x: x[1])
    sorted_infolist = sorted(sorted_infolist, key=lambda x: x[2])
    comm.Barrier()

    return rankinfo, sorted_infolist
