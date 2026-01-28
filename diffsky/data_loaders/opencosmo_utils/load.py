from __future__ import annotations

from pathlib import Path

import h5py
import opencosmo as oc

from ..hac_utils.lc_mock import (
    load_diffsky_param_collection,
    load_diffsky_ssp_data,
    load_diffsky_t_table,
    load_diffsky_tcurves,
)


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
    return catalog, aux_data


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
