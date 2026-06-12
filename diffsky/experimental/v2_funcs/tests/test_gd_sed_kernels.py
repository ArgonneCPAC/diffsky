""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.photometry import photometry_kernels as phk
from jax import random as jran
from jax import vmap

from ....param_utils import diffsky_param_wrapper_merging as dpwm
from ...tests import test_lightcone_generators as tlcg
from .. import gd_phot_kernels, gd_sed_kernels

_A = [None, 0, None, None, 0, *[None] * 4]
calc_obs_mags_galpop = vmap(phk.calc_obs_mag, in_axes=_A)


def test_sed_kern(num_halos=5):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos, n_z_phot_table=15
    )

    fb = 0.116
    ran_key, phot_key = jran.split(ran_key, 2)

    upid = np.where(lc_data.is_central == 1, -1, lc_data.halo_indx)
    lgmu_infall = lc_data.logmp_infall - lc_data.logmhost_infall
    gyr_since_infall = lc_data.t_obs - lc_data.t_infall
    phot_kern_results, phot_randoms, diffstarpop_results = (
        gd_phot_kernels._mc_phot_kern(
            ran_key,
            lc_data.z_obs,
            lc_data.t_obs,
            lc_data.mah_params,
            upid,
            lgmu_infall,
            lc_data.logmhost_infall,
            gyr_since_infall,
            lc_data.ssp_data,
            lc_data.precomputed_ssp_mag_table,
            lc_data.z_phot_table,
            lc_data.wave_eff_table,
            *dpwm.DEFAULT_PARAM_COLLECTION,
            DEFAULT_COSMOLOGY,
            fb,
        )
    )

    sed_kern_results = gd_sed_kernels._sed_kern(
        phot_randoms,
        diffstarpop_results.sfh_params,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        upid,
        lgmu_infall,
        lc_data.logmhost_infall,
        gyr_since_infall,
        lc_data.ssp_data,
        *dpwm.DEFAULT_PARAM_COLLECTION[1:],
        DEFAULT_COSMOLOGY,
        fb,
    )
    rest_sed_recomputed = sed_kern_results[0]

    n_met = lc_data.ssp_data.ssp_lgmet.size
    assert sed_kern_results.lgmet_weights.shape[1] == n_met

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

        mags = calc_obs_mags_galpop(*args)
        assert np.allclose(mags, phot_kern_results.obs_mags[:, iband], rtol=0.01)
