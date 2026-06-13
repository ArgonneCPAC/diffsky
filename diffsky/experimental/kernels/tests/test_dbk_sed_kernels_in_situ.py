""""""

import numpy as np
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.photometry import photometry_kernels as phk
from jax import random as jran
from jax import vmap

from ....param_utils import diffsky_param_wrapper_merging as dpwm
from ...tests import test_lightcone_generators as tlcg
from .. import dbk_sed_kernels_in_situ, mc_randoms, phot_kernels_in_situ, sed_kernels_in_situ

_A = [None, 0, None, None, 0, *[None] * 4]
calc_obs_mags_galpop = vmap(phk.calc_obs_mag, in_axes=_A)


def test_dbk_sed_kern(num_halos=20):
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.176

    n_gals = lc_data.z_obs.size
    phot_key, dbk_key = jran.split(ran_key, 2)
    dbk_randoms = mc_randoms.get_mc_dbk_randoms(dbk_key, n_gals)

    upid = np.where(lc_data.is_central == 1, -1, lc_data.halo_indx).astype(int)
    lgmu_infall = lc_data.logmp_infall - lc_data.logmhost_infall
    gyr_since_infall = lc_data.t_obs - lc_data.t_infall
    phot_kern_results, phot_randoms, merging_randoms = phot_kernels_in_situ._mc_phot_kern(
        phot_key,
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

    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [getattr(phot_kern_results, key) for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    sed_kern_results = sed_kernels_in_situ._sed_kern(
        phot_randoms,
        sfh_params,
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

    dbk_sed_kern_results = dbk_sed_kernels_in_situ._dbk_sed_kern(
        phot_randoms,
        dbk_randoms,
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
        lc_data.halo_indx,
    )

    assert np.allclose(
        np.log10(sed_kern_results.rest_sed),
        np.log10(dbk_sed_kern_results.rest_sed),
        atol=0.1,
    )

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
            dbk_sed_kern_results.rest_sed,
            lc_data.ssp_data.ssp_wave,
            trans_iband,
            lc_data.z_obs,
            *DEFAULT_COSMOLOGY,
        )

        mags = calc_obs_mags_galpop(*args)
        assert np.allclose(mags, phot_kern_results.obs_mags[:, iband], rtol=0.01)

    # Enforce agreement between composite SED and sum of component SEDs
    log_sed_composite = np.log10(dbk_sed_kern_results.rest_sed)
    log_sed_sum = np.log10(
        dbk_sed_kern_results.rest_sed_bulge
        + dbk_sed_kern_results.rest_sed_disk
        + dbk_sed_kern_results.rest_sed_knots
    )
    assert np.allclose(log_sed_composite, log_sed_sum, atol=0.1)

    # Enforce component masses sum to composite mass
    mstar_sum = (
        dbk_sed_kern_results.mstar_bulge
        + dbk_sed_kern_results.mstar_disk
        + dbk_sed_kern_results.mstar_knots
    )
    assert np.allclose(np.log10(mstar_sum), dbk_sed_kern_results.logsm_obs, atol=0.01)
