""""""

import numpy as np
import pytest
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.photometry import photometry_kernels as phk
from jax import random as jran
from jax import vmap

from ....param_utils import diffsky_param_wrapper_merging as dpwm
from ...tests import test_lightcone_generators as tlcg
from .. import phot_kernels_merging as gd_pkm
from .. import sed_kernels_merging as gd_sedkm

_A = [None, 0, None, None, 0, *[None] * 4]
calc_obs_mags_galpop = vmap(phk.calc_obs_mag, in_axes=_A)


@pytest.mark.parametrize("mc_merge", [0, 1])
def test_sed_kern(mc_merge, num_halos=5, return_results=False):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.176

    phot_kern_results, phot_randoms, merging_randoms = gd_pkm._mc_phot_kern_merging(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpwm.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.sat_weight,
        lc_data.halo_indx,
        mc_merge,
    )

    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [getattr(phot_kern_results, key) for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    sed_kern_results = gd_sedkm._sed_kern(
        phot_randoms,
        merging_randoms,
        sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        *dpwm.DEFAULT_PARAM_COLLECTION[1:],
        DEFAULT_COSMOLOGY,
        fb,
        lc_data.logmp_infall,
        lc_data.logmhost_infall,
        lc_data.t_infall,
        lc_data.is_central,
        lc_data.sat_weight,
        lc_data.halo_indx,
        mc_merge,
    )
    rest_sed_recomputed = sed_kern_results.rest_sed

    if return_results:
        return sed_kern_results, phot_kern_results, lc_data, tcurves

    # Enforce agreement between precomputed vs exact magnitudes
    n_bands = phot_kern_results.obs_mags.shape[1]
    for iband in range(n_bands):
        trans_iband = np.interp(
            lc_data.ssp_data.ssp_wave,
            tcurves[iband].wave,
            tcurves[iband].transmission,
        )
        args = (
            lc_data.ssp_data.ssp_wave,
            rest_sed_recomputed,
            lc_data.ssp_data.ssp_wave,
            trans_iband,
            lc_data.z_obs,
            *DEFAULT_COSMOLOGY,
        )

        recomputed_obs_mags = calc_obs_mags_galpop(*args)
        n_nan_cens = np.sum(~np.isfinite(recomputed_obs_mags[lc_data.is_central == 1]))
        n_nan_sats = np.sum(~np.isfinite(recomputed_obs_mags[lc_data.is_central == 0]))

        dmag = recomputed_obs_mags - phot_kern_results.obs_mags[:, iband]

        assert n_nan_sats == 0, "Some sats have NaN recomputed photometry"
        msg = "Discrepancy in recomputed photometry for sats"
        assert np.allclose(dmag[lc_data.is_central == 0], 0.0, atol=0.1), msg

        assert n_nan_cens == 0, "Some cens have NaN recomputed photometry"
        msg = "Discrepancy in recomputed photometry for cens"
        assert np.allclose(dmag[lc_data.is_central == 1], 0.0, atol=0.1), msg
