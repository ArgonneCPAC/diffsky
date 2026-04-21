""" """

import numpy as np
import pytest
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_emline_info as lemi
from jax import random as jran

from ....merging import merging_model
from ....param_utils import diffsky_param_wrapper as dpw
from ...tests import test_lightcone_generators as tlcg
from .. import specphot_kernels_merging as sppkm
from .test_phot_kernels_merging import check_phot_kern_merging_results

TOL = 1e-8


def check_spec_kern_merging_results(spec_kern_results, lc_data):
    n_gals = lc_data.z_obs.size
    n_lines = len(lc_data.ssp_data.ssp_emline_wave)

    assert spec_kern_results.linelum_gal.shape == (n_gals, n_lines)
    assert spec_kern_results.linelum_gal_in_situ.shape == (n_gals, n_lines)

    for arr in spec_kern_results:
        assert np.all(np.isfinite(arr))

    msk_cen = lc_data.is_central == 1
    msk_sat = ~msk_cen

    # Enforce centrals can only get more massive and satellites less massive
    name = "linelum_gal"
    x = np.log10(getattr(spec_kern_results, name))
    y = np.log10(getattr(spec_kern_results, name + "_in_situ"))
    assert np.all(x[msk_cen] >= y[msk_cen] - TOL)
    assert np.all(x[msk_sat] <= y[msk_sat] + TOL)
    # Enforce merging is nontrivial
    assert np.any(x[msk_cen] > y[msk_cen])
    assert np.any(x[msk_sat] < y[msk_sat])


@pytest.mark.parametrize("mc_merge", [0, 1])
def test_mc_specphot_kern_merging(mc_merge, num_halos=141):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.151

    n_lines = 3
    line_wave_table = np.linspace(1_000, 10_000, n_lines)
    emline_names = lc_data.ssp_data.ssp_emline_wave._fields[0:n_lines]
    ssp_data = lemi.get_subset_emline_data(lc_data.ssp_data, emline_names)
    lc_data = lc_data._replace(ssp_data=ssp_data)

    phot_kern_results, phot_randoms, spec_kern_results = (
        sppkm._mc_specphot_kern_merging(
            ran_key,
            lc_data.z_obs,
            lc_data.t_obs,
            lc_data.mah_params,
            lc_data.ssp_data,
            lc_data.precomputed_ssp_mag_table,
            lc_data.z_phot_table,
            lc_data.wave_eff_table,
            line_wave_table,
            *dpw.DEFAULT_PARAM_COLLECTION,
            merging_model.DEFAULT_MERGE_PARAMS,
            DEFAULT_COSMOLOGY,
            fb,
            lc_data.logmp_infall,
            lc_data.logmhost_infall,
            lc_data.t_infall,
            lc_data.is_central,
            lc_data.nhalos,
            lc_data.halo_indx,
            mc_merge,
        )
    )

    check_phot_kern_merging_results(phot_kern_results, lc_data)
    check_spec_kern_merging_results(spec_kern_results, lc_data)
