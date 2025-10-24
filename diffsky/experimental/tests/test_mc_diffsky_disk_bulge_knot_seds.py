""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.photometry import photometry_kernels as phk
from jax import random as jran
from jax import vmap

from ...param_utils import diffsky_param_wrapper as dpw
from .. import lc_phot_kern
from .. import mc_diffsky_disk_bulge_knot_seds as mcsed_dbk
from ..disk_bulge_modeling import disk_knots
from . import test_lc_phot_kern as tlcphk

_A = [None, 0, None, None, 0, *[None] * 4]
calc_obs_mags_galpop = vmap(phk.calc_obs_mag, in_axes=_A)


def test_mc_diffsky_seds_flat_u_params():
    ran_key = jran.key(0)
    lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing()

    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape
    assert lc_data.z_phot_table.shape == (n_z_table,)
    assert lc_data.ssp_data.ssp_lgmet.shape == (n_met,)
    assert lc_data.ssp_data.ssp_lg_age_gyr.shape == (n_age,)

    u_param_collection = dpw.get_u_param_collection_from_param_collection(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    u_param_arr = dpw.unroll_u_param_collection_into_flat_array(*u_param_collection)

    sed_info = mcsed_dbk._mc_diffsky_seds_dbk_flat_u_params(
        u_param_arr, ran_key, lc_data, DEFAULT_COSMOLOGY
    )
    lc_phot_orig = lc_phot_kern.multiband_lc_phot_kern_u_param_arr(
        u_param_arr, ran_key, lc_data
    )
    msk_q = sed_info["mc_sfh_type"] == 0
    assert np.allclose(lc_phot_orig.obs_mags_q[msk_q], sed_info["obs_mags"][msk_q])
    msk_smooth_ms = sed_info["mc_sfh_type"] == 1
    assert np.allclose(
        lc_phot_orig.obs_mags_smooth_ms[msk_smooth_ms],
        sed_info["obs_mags"][msk_smooth_ms],
    )
    msk_bursty_ms = sed_info["mc_sfh_type"] == 2
    assert np.allclose(
        lc_phot_orig.obs_mags_bursty_ms[msk_bursty_ms],
        sed_info["obs_mags"][msk_bursty_ms],
    )

    # check rest_sed_bulge
    assert sed_info["rest_sed"].shape == sed_info["rest_sed_bulge"].shape
    assert np.all(np.isfinite(sed_info["rest_sed_bulge"]))
    assert not np.allclose(sed_info["rest_sed_bulge"], sed_info["rest_sed"], rtol=0.1)
    assert np.all(sed_info["rest_sed_bulge"] >= 0)
    assert np.any(sed_info["rest_sed_bulge"] > 0)

    # check rest_sed_disk
    assert sed_info["rest_sed"].shape == sed_info["rest_sed_disk"].shape
    assert np.all(np.isfinite(sed_info["rest_sed_disk"]))
    assert not np.allclose(sed_info["rest_sed_disk"], sed_info["rest_sed"], rtol=0.1)
    assert not np.allclose(
        sed_info["rest_sed_disk"], sed_info["rest_sed_bulge"], rtol=0.1
    )
    assert np.all(sed_info["rest_sed_disk"] >= 0)
    assert np.any(sed_info["rest_sed_disk"] > 0)

    # check rest_sed_knot
    assert sed_info["rest_sed"].shape == sed_info["rest_sed_knot"].shape
    assert np.all(np.isfinite(sed_info["rest_sed_knot"]))
    assert not np.allclose(sed_info["rest_sed_knot"], sed_info["rest_sed"], rtol=0.1)
    assert not np.allclose(
        sed_info["rest_sed_knot"], sed_info["rest_sed_bulge"], rtol=0.1
    )
    assert np.all(sed_info["rest_sed_knot"] >= 0)
    assert np.any(sed_info["rest_sed_knot"] > 0)

    assert np.all(np.isfinite(sed_info["fknot"]))
    assert np.all(sed_info["fknot"] > 0)
    assert np.all(sed_info["fknot"] < disk_knots.FKNOT_MAX)

    assert np.all(sed_info["rest_sed_bulge"] < sed_info["rest_sed"])
    assert np.all(sed_info["rest_sed_disk"] < sed_info["rest_sed"])

    # Allow knots to be slightly brighter than the original SED
    # The original SED, after all, is not strictly decomposed into components
    # This rarely happens anyway, except for ~2% of galaxies
    # and even then only for UV wavelengths
    msk_uv_wave = lc_data.ssp_data.ssp_wave < 2_000
    msk_optical_wave = lc_data.ssp_data.ssp_wave > 4_000
    n_gals = sed_info["rest_sed"].shape[0]
    for igal in range(n_gals):
        assert np.all(
            sed_info["rest_sed_knot"][igal, :][msk_optical_wave]
            < sed_info["rest_sed"][igal, :][msk_optical_wave] * 1.1
        )
        assert np.all(
            sed_info["rest_sed_knot"][igal, :][msk_uv_wave]
            < sed_info["rest_sed"][igal, :][msk_uv_wave] * 1.5
        )

    # _check_sed_info(sed_info, lc_data, tcurves)

    assert np.all(np.isfinite(sed_info["obs_mags_bulge"]))
    assert np.all(np.isfinite(sed_info["obs_mags_disk"]))
    assert np.all(np.isfinite(sed_info["obs_mags_knots"]))

    assert not np.allclose(sed_info["obs_mags"], sed_info["obs_mags_bulge"], rtol=1e-4)
    assert np.all(sed_info["obs_mags"] <= sed_info["obs_mags_bulge"])

    assert not np.allclose(sed_info["obs_mags"], sed_info["obs_mags_disk"], rtol=1e-4)
    assert np.all(sed_info["obs_mags"] <= sed_info["obs_mags_disk"])

    assert not np.allclose(sed_info["obs_mags"], sed_info["obs_mags_knots"], rtol=1e-4)
    assert np.all(sed_info["obs_mags"] <= sed_info["obs_mags_knots"])


def test_mc_diffsky_phot_dbk_flat_u_params():
    """Enforce _mc_diffsky_seds_dbk_flat_u_params is consistent with _mc_diffsky_phot_flat_u_params"""
    ran_key = jran.key(0)
    lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing()

    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape
    assert lc_data.z_phot_table.shape == (n_z_table,)
    assert lc_data.ssp_data.ssp_lgmet.shape == (n_met,)
    assert lc_data.ssp_data.ssp_lg_age_gyr.shape == (n_age,)

    u_param_collection = dpw.get_u_param_collection_from_param_collection(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    u_param_arr = dpw.unroll_u_param_collection_into_flat_array(*u_param_collection)

    sed_info = mcsed_dbk._mc_diffsky_seds_dbk_flat_u_params(
        u_param_arr, ran_key, lc_data, DEFAULT_COSMOLOGY
    )
    phot_info = mcsed_dbk._mc_diffsky_phot_dbk_flat_u_params(
        u_param_arr, ran_key, lc_data, DEFAULT_COSMOLOGY
    )

    assert np.allclose(sed_info["obs_mags"], phot_info["obs_mags"], rtol=1e-4)

    assert sed_info["obs_mags"].shape == phot_info["obs_mags_bulge"].shape

    for key in ("obs_mags", "obs_mags_bulge", "obs_mags_disk", "obs_mags_knots"):
        assert np.allclose(sed_info[key], phot_info[key], rtol=1e-4)
