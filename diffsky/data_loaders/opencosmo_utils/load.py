from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np
import opencosmo as oc

from ..hacc_utils.lc_mock import (
    load_diffsky_param_collection,
    load_diffsky_ssp_data,
    load_diffsky_t_table,
    load_diffsky_tcurves,
)
from .versions import check_versions


def load_diffsky_mock(
    path: str | Path, synth_cores: bool = False
) -> tuple[oc.Lightcone, dict | None]:
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise NotADirectoryError(path)

    files = list(path.glob("*.hdf5"))
    data_files = list(
        filter(lambda p: p.stem.endswith("diffsky_gals"), files)
    )  # Not super robust, should come up with something better

    if not data_files:
        raise FileNotFoundError(f"Found no data files in directory {str(path)}")

    with h5py.File(data_files[0]) as f:
        mock_name = f["header"]["catalog_info"].attrs[
            "mock_version_name"
        ]  # in files, but not currently exposed by the toolkit

    # Not a failure case to not have auxillary data for loading purposes.
    # users may wabnt to manipulate the data itself for other reasons.
    try:
        aux_data = __load_aux_data(path, mock_name)
    except FileNotFoundError:
        aux_data = None
    catalog = oc.open(data_files, synth_cores=synth_cores)
    check_versions(catalog)

    return catalog, aux_data


def add_transmission_curves(
    aux_data: dict, **transmission_curves: tuple[np.ndarray, np.ndarray]
):
    """
    Add new tranmission curves to the diffsky metadata. These new curves can then be
    used to compute photometry.

    """
    TransmissionCurve = namedtuple("TransmissionCurve", ("wave", "transmission"))
    known_tcurves = aux_data["tcurves"]._fields

    overwrites = set(known_tcurves).intersection(transmission_curves.keys())
    if overwrites:
        raise ValueError(
            f"Tried to add transmission curves that already exist: {overwrites}"
        )

    new_curves = {}
    for name, curve in transmission_curves.items():
        wave = curve[0]
        tcurve = curve[1]
        if not np.all(wave[1:] > wave[:-1]):
            raise ValueError(
                "Transmission curve wavelength specifications should always go from low to high"
            )
        new_curves[name] = TransmissionCurve(wave, tcurve)

    new_filter_nicknames = (*known_tcurves, *tuple(new_curves.keys()))
    TCurves = namedtuple("TCurves", new_filter_nicknames)
    new_tcurves = TCurves(*aux_data["tcurves"], *tuple(new_curves.values()))
    new_aux_data = {**aux_data, "tcurves": new_tcurves}
    return new_aux_data


def __load_aux_data(path: Path, mock_name: str):
    tcurves = load_diffsky_tcurves(
        path, mock_name
    )  # note pathlib.Path is compatible with os.path calls
    t_table = load_diffsky_t_table(path, mock_name)
    ssp_data = load_diffsky_ssp_data(path, mock_name)
    param_collection = load_diffsky_param_collection(path, mock_name)

    diffsky_aux_data = dict(
        ssp_data=ssp_data,
        param_collection=param_collection,
        tcurves=tcurves,
        t_table=t_table,
    )
    return diffsky_aux_data
