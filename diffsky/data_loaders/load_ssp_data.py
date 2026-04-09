""""""

from collections import namedtuple

import h5py
from dsps.constants import L_SUN_CGS
from dsps.data_loaders import load_ssp_templates as load_ssp_templates_dsps
from dsps.data_loaders.load_emline_info import get_subset_emline_data
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


def load_fake_ssp_data():
    ssp_data = load_fake_ssp_data_dsps()
    return ssp_data


def get_sparse_ssp_data(
    ssp_data,
    n_met=5,
    n_age=7,
    n_wave=90,
    emline_names=("Ba_alpha_6563", "Ba_beta_4861"),
):
    """Get a tiny subset of the ssp_data in each dimension - mostly for unit-testing"""
    if "ssp_emline_wave" in ssp_data._fields:
        ssp_data = get_subset_emline_data(ssp_data, emline_names)

    n_skip_met = ssp_data.ssp_lgmet.size // n_met
    lgmet_sparse = ssp_data.ssp_lgmet[::n_skip_met]

    n_skip_lg_age_gyr = ssp_data.ssp_lg_age_gyr.size // n_age
    lg_age_gyr_sparse = ssp_data.ssp_lg_age_gyr[::n_skip_lg_age_gyr]

    n_skip_wave = ssp_data.ssp_wave.size // n_wave
    wave_sparse = ssp_data.ssp_wave[::n_skip_wave]

    ssp_flux = ssp_data.ssp_flux[::n_skip_met, ::n_skip_lg_age_gyr, ::n_skip_wave]

    emline_lum_sparse = ssp_data.ssp_emline_luminosity[
        ::n_skip_met, ::n_skip_lg_age_gyr
    ]

    sparse_ssp_data = ssp_data._replace(
        ssp_lgmet=lgmet_sparse,
        ssp_lg_age_gyr=lg_age_gyr_sparse,
        ssp_wave=wave_sparse,
        ssp_flux=ssp_flux,
        ssp_emline_luminosity=emline_lum_sparse,
    )
    return sparse_ssp_data
