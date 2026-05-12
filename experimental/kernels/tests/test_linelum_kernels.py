""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_emline_info as lemi
from jax import random as jran

from ....param_utils import diffsky_param_wrapper as dpw
from ...tests import test_mc_lightcone_halos as tmclh
from .. import linelum_kernels, phot_kernels


def test_mc_specphot_kern(num_halos=150):
    ran_key = jran.key(0)
    lc_data, tcurves = tmclh._get_weighted_lc_data_for_unit_testing(num_halos=num_halos)

    fb = 0.156
    ran_key, phot_key = jran.split(ran_key, 2)

    phot_kern_results, phot_randoms = phot_kernels._mc_phot_kern(
        phot_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpw.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )

    n_lines = 3
    line_wave_table = np.linspace(1_000, 10_000, n_lines)
    emline_names = lc_data.ssp_data.ssp_emline_wave._fields[0:n_lines]
    ssp_data = lemi.get_subset_emline_data(lc_data.ssp_data, emline_names)
    lc_data = lc_data._replace(ssp_data=ssp_data)

    _specphot_res = linelum_kernels._mc_specphot_kern(
        phot_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        line_wave_table,
        *dpw.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )

    phot_kern_results2, phot_randoms2, spec_kern_results = _specphot_res
    assert np.allclose(
        phot_kern_results.obs_mags, phot_kern_results2.obs_mags, rtol=1e-4
    )
