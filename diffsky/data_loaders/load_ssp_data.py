""""""

import os
import random
import string
from collections import OrderedDict, namedtuple

import h5py
import numpy as np
from dsps.data_loaders.defaults import SSPData as DEFAULT_SSPData
from dsps.data_loaders.retrieve_fake_fsps_data import (
    load_fake_ssp_data as load_fake_ssp_data_dsps,
)

from ..utils.emline_utils import L_SUN_CGS

EmissionLine = namedtuple("EmissionLine", ["line_wave", "line_flux"])

DEFAULT_DIFFSKY_SSP_BNAME = "ssp_data_fsps_v3.2_emlines.hdf5"


def load_ssp_templates(fn=None, drn=None, bn=DEFAULT_DIFFSKY_SSP_BNAME):
    """Load SSP templates, optionally including emission lines if present

    For emission lines, note that line_flux is stored on disk in units of Lsun/Msun.
    But load_ssp_templates converts to cgs units after reading from disk.
    And write_ssp_templates_to_disk converts back to Lsun/Msun before writing to disk.

    """

    if fn is None:
        if drn is None:
            try:
                drn = os.environ["DSPS_DRN"]
            except KeyError:
                msg = (
                    "Since you did not pass the fn or drn argument\n"
                    "then you must have the DSPS_DRN environment variable set"
                )
                raise ValueError(msg)

        fn = os.path.join(drn, bn)

    msg = "{0} does not exist".format(fn)
    assert os.path.isfile(fn), msg

    EmissionLine = namedtuple("EmissionLine", ["line_wave", "line_flux"])

    ssp_data_dict = OrderedDict()

    with h5py.File(fn, "r") as hdf:
        for key in DEFAULT_SSPData._fields:
            ssp_data_dict[key] = hdf[key][...]

        if "emlines" in hdf.keys():
            emlines_dict = OrderedDict()
            for emline in hdf["emlines"].keys():
                wave = float(hdf["emlines"][emline]["line_wave"][...])
                flux = hdf["emlines"][emline]["line_flux"][...] * L_SUN_CGS
                emlines_dict[emline] = EmissionLine(wave, flux)

            EmissionLines = namedtuple("EmissionLines", list(emlines_dict.keys()))
            emission_lines = EmissionLines(**emlines_dict)
            ssp_data_dict["emlines"] = emission_lines

    SSPData = namedtuple("SSPData", list(ssp_data_dict.keys()))
    ssp_data = SSPData(**ssp_data_dict)

    return ssp_data


def write_ssp_templates_to_disk(fn, ssp_data):
    """Write the SSP data to disk

    For emission lines, note that line_flux is stored on disk in units of Lsun/Msun.
    But load_ssp_templates converts to cgs units after reading from disk.
    And write_ssp_templates_to_disk converts back to Lsun/Msun before writing to disk.

    """

    with h5py.File(fn, "w") as hdf_out:
        for name, arr in zip(ssp_data._fields, ssp_data):
            if name != "emlines":
                hdf_out[name] = arr

        if "emlines" in ssp_data._fields:
            grp = hdf_out.create_group("emlines")

            # Store each line's wavelength and flux table
            gen = zip(ssp_data.emlines._fields, ssp_data.emlines)
            for line_name, emline in gen:
                line_grp = grp.create_group(line_name)
                line_grp["line_wave"] = emline.line_wave
                line_grp["line_flux"] = emline.line_flux / L_SUN_CGS


def load_fake_ssp_data(n_lines=3, emline_names=None):
    ssp_data = load_fake_ssp_data_dsps()
    n_met, n_age = ssp_data.ssp_flux.shape[:-1]

    line_waves = np.linspace(1_000, 10_000, n_lines)
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
