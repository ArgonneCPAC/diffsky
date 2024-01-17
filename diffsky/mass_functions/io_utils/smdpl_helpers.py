"""
"""
import numpy as np

from .loader import um_dtype

SMDPL_LGMP = np.log10(1.5e8)


def load_smdpl_binary(fname):
    """Load a SMDPL UniverseMachine binary and precompute a few quantities
    needed to tabulate the CCSHMF
    """
    halos = np.fromfile(fname, dtype=um_dtype)
    from halotools.utils import crossmatch

    cenmsk = halos["upid"] == -1
    hostid = np.where(cenmsk, halos["id"], halos["upid"])

    idxA, idxB = crossmatch(hostid, halos["id"])

    assert len(idxA) == len(halos["id"]), "unmatched halos in {}".format(fname)

    from astropy.table import Table

    mock = Table()

    mock["halo_id"] = halos["id"]
    mock["hostid"] = hostid
    mock["logmp"] = np.log10(halos["mp"])

    host_mpeak = np.copy(halos["mp"])
    host_mpeak[idxA] = halos["mp"][idxB]
    mock["host_lgmp"] = np.log10(host_mpeak)
    mock["lgmu_peak"] = np.log10(halos["mp"] / host_mpeak)

    del halos
    return mock
