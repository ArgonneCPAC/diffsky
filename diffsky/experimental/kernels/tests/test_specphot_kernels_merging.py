""" """

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_emline_info as lemi
from jax import random as jran

from ....merging import merging_model
from ....param_utils import diffsky_param_wrapper as dpw
from ...tests import test_lightcone_generators as tlcg
from .. import mc_phot_kernels as mcpk
from .. import specphot_kernels_merging as sppkm


def test_specphot_kern_merging(num_halos=250):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )

    fb = 0.156
    ran_key, phot_key = jran.split(ran_key, 2)

    phot_randoms, sfh_params = mcpk.get_mc_phot_randoms(
        ran_key, dpw.DEFAULT_PARAM_COLLECTION[0], lc_data.mah_params, DEFAULT_COSMOLOGY
    )

    n_lines = 3
    line_wave_table = np.linspace(1_000, 10_000, n_lines)
    emline_names = lc_data.ssp_data.ssp_emline_wave._fields[0:n_lines]
    ssp_data = lemi.get_subset_emline_data(lc_data.ssp_data, emline_names)
    lc_data = lc_data._replace(ssp_data=ssp_data)

    (
        phot_kern_results,
        linelums_in_situ,
        flux_obs,
        merge_prob,
        mstar_obs,
        linelums_in_plus_ex_situ,
    ) = sppkm._specphot_kern_merging(
        phot_randoms,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        line_wave_table,
        *dpw.DEFAULT_PARAM_COLLECTION[1:],
        merging_model.DEFAULT_MERGE_PARAMS,
        DEFAULT_COSMOLOGY,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.nhalos,
        lc_data.halo_indx,
    )

    assert np.all(merge_prob >= 0)
    assert np.all(merge_prob <= 1)
    assert np.any(merge_prob > 0)
    assert np.any(merge_prob < 1)

    assert np.all(np.isfinite(mstar_obs))

    assert np.any(linelums_in_plus_ex_situ != linelums_in_situ)
