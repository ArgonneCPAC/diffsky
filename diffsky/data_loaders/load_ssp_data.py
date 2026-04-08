""""""

from collections import namedtuple

import h5py
from dsps.constants import L_SUN_CGS
from dsps.data_loaders import load_ssp_templates as load_ssp_templates_dsps
from dsps.data_loaders.retrieve_fake_fsps_data import (
    load_fake_ssp_data as load_fake_ssp_data_dsps,
)

from .defaults import DEFAULT_SSP_BNAME

EmissionLine = namedtuple("EmissionLine", ["line_wave", "line_flux"])


def load_ssp_templates(fn=None, drn=None, bn=DEFAULT_SSP_BNAME):
    ssp_data = load_ssp_templates_dsps(fn=fn, drn=drn, bn=bn)
    return ssp_data


load_ssp_templates.__doc__ = load_ssp_templates_dsps.__doc__


def write_ssp_templates_to_disk(fn, ssp_data):
    """Write the SSP data to disk

    For emission lines, note that line_flux is stored on disk in units of Lsun/Msun.
    But load_ssp_templates converts to cgs units after reading from disk.
    And write_ssp_templates_to_disk converts back to Lsun/Msun before writing to disk.

    """

    with h5py.File(fn, "w") as hdf_out:
        for field, val in zip(ssp_data._fields, ssp_data):
            if val is not None:
                if field == "ssp_emline_luminosity":
                    hdf_out[field] = val / L_SUN_CGS
                else:
                    hdf_out[field] = val

        if "ssp_emline_wave" in ssp_data._fields:
            hdf_out["ssp_emline_name"] = list(ssp_data.ssp_emline_wave._fields)


def load_fake_ssp_data(n_lines=3, emline_names=None):
    ssp_data = load_fake_ssp_data_dsps()
    return ssp_data
