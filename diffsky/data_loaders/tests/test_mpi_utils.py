""""""

import numpy as np

from .. import mpi_utils


def test_distribute_files_by_size():
    nranks = 2
    sizes = np.zeros(nranks) + 0.1
    rank_assignments, rank_totals = mpi_utils.distribute_files_by_size(sizes, nranks)
    assert len(rank_assignments) == nranks
    assert np.all(rank_totals > 0)


def test_distribute_files_by_size_edge_case():
    nranks = 1
    sizes = np.zeros(nranks) + 0.1
    rank_assignments, rank_totals = mpi_utils.distribute_files_by_size(sizes, nranks)
    assert len(rank_assignments) == nranks
    assert np.all(rank_totals > 0)
