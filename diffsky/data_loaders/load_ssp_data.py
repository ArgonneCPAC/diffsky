""""""

import random
import string
from collections import namedtuple

import h5py
import numpy as np
from dsps.data_loaders import load_ssp_templates as load_ssp_templates_dsps
from dsps.data_loaders.retrieve_fake_fsps_data import (
    load_fake_ssp_data as load_fake_ssp_data_dsps,
)

from ..utils.emline_utils import L_SUN_CGS
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
                hdf_out[field] = val

        if "ssp_emline_wave" in ssp_data._fields:
            hdf_out["ssp_emline_name"] = list(ssp_data.ssp_emline_wave._fields)


def load_fake_ssp_data(n_lines=3, emline_names=None):
    ssp_data = load_fake_ssp_data_dsps()
    n_met, n_age = ssp_data.ssp_flux.shape[:-1]

    characters = string.ascii_letters

    if emline_names is None:
        linename_length = 10
        emline_names = []
        while len(emline_names) < n_lines:
            random_string = "".join(random.choices(characters, k=linename_length))
            if random_string not in emline_names:
                emline_names.append(random_string)
    else:
        n_lines = len(emline_names)

    line_waves = np.linspace(1_000, 10_000, n_lines)

    emlines_dict = dict()
    for i, linename in enumerate(emline_names):
        line_wave = line_waves[i]
        line_flux = np.ones((n_met, n_age))
        emlines_dict[linename] = EmissionLine(line_wave, line_flux)

    EmissionLines = namedtuple("EmissionLines", list(emlines_dict.keys()))
    emission_lines = EmissionLines(**emlines_dict)

    ssp_data_dict = ssp_data._asdict()
    ssp_data_dict["emlines"] = emission_lines

    SSPData = namedtuple("SSPData", list(ssp_data_dict.keys()))
    ssp_data = SSPData(**ssp_data_dict)

    return ssp_data
