""""""

import os
from collections import OrderedDict, namedtuple

import h5py
from dsps.data_loaders.defaults import SSPData as DEFAULT_SSPData

from ..utils.emline_utils import L_SUN_CGS

EmissionLine = namedtuple("EmissionLine", ["line_wave", "line_flux"])

DEFAULT_DIFFSKY_SSP_BNAME = "ssp_data_fsps_v3.2_emlines.hdf5"


def load_ssp_templates(fn=None, drn=None, bn=DEFAULT_DIFFSKY_SSP_BNAME):
    """Load SSP templates, optionally including emission lines if present"""

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
