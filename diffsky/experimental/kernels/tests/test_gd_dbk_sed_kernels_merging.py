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
from .. import gd_dbk_sed_kernels_merging as gd_dbk_sedkm
from .. import gd_dbk_specphot_kernels_merging
from .. import gd_sed_kernels_merging as gd_sedkm
from .. import mc_randoms

_A = [None, 0, None, None, 0, *[None] * 4]
calc_obs_mags_galpop = vmap(phk.calc_obs_mag, in_axes=_A)


@pytest.mark.parametrize("z_med", [0.5, 2.5])
def test_sed_kern(
    z_med, mc_merge=1, num_halos=5, return_results=False, dz=0.01, ssp_data=None
):
    ran_key = jran.key(0)
    z_min, z_max = z_med - dz, z_med + dz
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos,
        z_min=z_min,
        z_max=z_max,
        n_z_phot_table=2,
        ssp_data=ssp_data,
    )
    fb = 0.176

    phot_randoms, sfh_params, dbk_randoms, merging_randoms = (
        mc_randoms.get_mc_dbk_phot_merge_randoms(
            ran_key,
            dpwm.DEFAULT_PARAM_COLLECTION.diffstarpop_params,
            lc_data.mah_params,
            DEFAULT_COSMOLOGY,
        )
    )

    args = (
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        lc_data.line_wave_table,
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
    _res = gd_dbk_specphot_kernels_merging._mc_dbk_specphot_kern_merging(*args)
    dbk_specphot_info, dbk_weights = _res

    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [getattr(dbk_specphot_info, key) for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    sed_info = gd_sedkm._sed_kern(
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

    dbk_sed_info = gd_dbk_sedkm._dbk_sed_kern(
        phot_randoms,
        dbk_randoms,
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

    ret = (
        lc_data,
        tcurves,
        dbk_specphot_info,
        phot_randoms,
        merging_randoms,
        dbk_sed_info,
        sed_info,
    )
    if return_results:
        return ret

    rest_sed_dbk_recomputed = (
        dbk_sed_info.rest_sed_bulge
        + dbk_sed_info.rest_sed_disk
        + dbk_sed_info.rest_sed_knots
    )

    msg = "Sum of DBK rest SEDs disagrees with composite SED"
    assert np.allclose(
        np.log10(rest_sed_dbk_recomputed), np.log10(sed_info.rest_sed), atol=0.1
    ), msg

    # Enforce agreement between gd_dbk_sed_kernels_merging vs gd_dbk_specphot_kernels_merging
    n_bands = dbk_specphot_info.obs_mags.shape[1]
    for iband in range(n_bands):
        trans_iband = np.interp(
            lc_data.ssp_data.ssp_wave,
            tcurves[iband].wave,
            tcurves[iband].transmission,
        )

        components = ("", "_bulge", "_disk", "_knots")
        for component in components:
            sed = getattr(dbk_sed_info, "rest_sed" + component)
            obs_mags = getattr(dbk_specphot_info, "obs_mags" + component)

            args = (
                lc_data.ssp_data.ssp_wave,
                sed,
                lc_data.ssp_data.ssp_wave,
                trans_iband,
                lc_data.z_obs,
                *DEFAULT_COSMOLOGY,
            )

            recomputed_obs_mags = calc_obs_mags_galpop(*args)
            specphot_obs_mags = obs_mags[:, iband]
            dmag = recomputed_obs_mags - specphot_obs_mags

            assert np.median(dmag) < 0.1, "Systematic offset in recomputed magnitudes"

            # compute trimmed variance
            dmag_lo, dmag_hi = np.percentile(dmag, (5, 95))
            msk_trim = (dmag > dmag_lo) & (dmag < dmag_hi)
            std_trim = np.std(dmag[msk_trim])
            assert std_trim < 0.1, "Large scatter in recomputed magnitudes"

            # compute outlier fraction
            DMAG_TOL = 1.0
            msk_outlier = np.abs(dmag) > DMAG_TOL
            num_outliers = np.sum(msk_outlier)
            ntot = dmag.size
            frac_outliers = num_outliers / ntot

            FOUT_TOL = 0.1
            msg = f"{num_outliers}/{ntot}>{FOUT_TOL:.2f} "
            msg += f"outliers with δmag>{DMAG_TOL:.2F} in approx-exact mags"
            assert frac_outliers < FOUT_TOL, msg
