""""""

from collections import namedtuple

import numpy as np
from dsps.data_loaders import load_emline_info as lemi
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.data_loaders.defaults import TransmissionCurve
from jax import random as jran

from ...data_loaders import load_ssp_data
from .. import lightcone_generators as lcg
from .. import mc_phot
from . import test_mc_phot as tmcp


def _get_weighted_lc_photdata_for_unit_testing(
    num_halos=75, n_lines=3, z_min=0.1, z_max=3.0, n_z_phot_table=15, ssp_data=None
):
    ran_key = jran.key(0)

    lgmp_min, lgmp_max = 10.0, 15.0
    sky_area_degsq = 100.0

    if ssp_data is None:
        ssp_data = load_ssp_data.load_fake_ssp_data()

    _res = retrieve_fake_fsps_data.load_fake_filter_transmission_curves()
    wave, u, g, r, i, z, y = _res

    tcurve_list = [TransmissionCurve(wave, x) for x in (u, i, y)]
    names = [f"lsst_{x}" for x in ("u", "i", "y")]
    TransmissionCurves = namedtuple("TransmissionCurves", names)
    tcurves = TransmissionCurves(*tcurve_list)

    z_phot_table = 10 ** np.linspace(np.log10(z_min), np.log10(z_max), n_z_phot_table)

    args = (
        ran_key,
        num_halos,
        z_min,
        z_max,
        lgmp_min,
        lgmp_max,
        sky_area_degsq,
        ssp_data,
        tcurves,
        z_phot_table,
    )
    lc_data = lcg.weighted_lc_photdata(*args)

    emline_names = lc_data.ssp_data.ssp_emline_wave._fields[0:n_lines]
    ssp_data = lemi.get_subset_emline_data(lc_data.ssp_data, emline_names)
    lc_data = lc_data._replace(
        ssp_data=ssp_data,
        line_wave_table=lc_data.line_wave_table[0:n_lines],
        precomputed_ssp_linelum_cgs_table=lc_data.precomputed_ssp_linelum_cgs_table[
            :n_lines, :, :
        ],
    )

    return lc_data, tcurves


def test_weighted_lc_photdata():
    num_halos = 75
    lc_data, tcurves = _get_weighted_lc_photdata_for_unit_testing(num_halos=num_halos)
    n_tot = lc_data.z_obs.size
    shape_ntot_keys = (
        "z_obs",
        "t_obs",
        "logmp_obs",
        "logmp0",
        "t_infall",
        "logmp_infall",
        "logmhost_infall",
        "is_central",
        "halo_indx",
    )
    for lc_key in shape_ntot_keys:
        arr = getattr(lc_data, lc_key)
        assert arr.shape == (n_tot,), f"lc_data.{lc_key} has the wrong shape"
        assert np.all(np.isfinite(arr)), f"lc_data.{lc_key} has NaNs"

    for field in lcg.LCData._fields:
        assert hasattr(lc_data, field)

    for arr in lc_data.mah_params:
        assert np.all(np.isfinite(arr))

    ran_key = jran.key(1)
    mc_merge = 0
    phot_kern_results = mc_phot.mc_lc_phot(ran_key, lc_data, mc_merge)[0]
    tmcp.check_phot_kern_results(phot_kern_results)
