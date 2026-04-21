""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ....param_utils import diffsky_param_wrapper_merging as dpwm
from ...tests import test_lightcone_generators as tlcg
from .. import dbk_specphot_kernels_merging as dbkspkm


def test_mc_dbk_specphot_kern_merging(num_halos=150, mc_merge=0):
    """Enforce that the sum of the component lines equals the composite line"""
    ran_key = jran.key(0)
    lc_data, tcurves = tlcg._get_weighted_lc_photdata_for_unit_testing(
        num_halos=num_halos
    )
    fb = 0.13

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
        lc_data.nhalos,
        lc_data.halo_indx,
        mc_merge,
    )
    _res = dbkspkm._mc_dbk_specphot_kern_merging(*args)
    dbk_specphot_info, dbk_weights = _res
    # return dbk_specphot_info, dbk_weights, lc_data
    for arr in dbk_specphot_info:
        assert np.all(np.isfinite(arr))

    component_names = ("_bulge", "_disk", "_knots")

    # Enforce consistent array shapes
    correct_shape = getattr(dbk_specphot_info, "logsm_obs").shape
    for k in component_names:
        kname = "logsm" + k
        component_shape = getattr(dbk_specphot_info, kname).shape
        assert correct_shape == component_shape, kname

    correct_shape = getattr(dbk_specphot_info, "obs_mags").shape
    for k in component_names:
        kname = "obs_mags" + k
        component_shape = getattr(dbk_specphot_info, kname).shape
        assert correct_shape == component_shape, kname

    correct_shape = getattr(dbk_specphot_info, "linelum_gal").shape
    for k in component_names:
        kname = "linelum" + k
        component_shape = getattr(dbk_specphot_info, kname).shape
        assert correct_shape == component_shape, kname

    # Enforce consistency with dbk_weights
    specphot_key, dbk_key = "logsm", "mstar"
    for k in component_names:
        x = getattr(dbk_specphot_info, specphot_key + k)
        y = np.log10(getattr(dbk_weights, dbk_key + k))
        assert np.allclose(x, y, atol=0.01)

    msk_cen = lc_data.is_central == 1
    msk_sat = ~msk_cen

    # Enforce centrals can only get more massive and satellites less massive
    name = "logsm_obs"
    x = getattr(dbk_specphot_info, name)
    y = getattr(dbk_specphot_info, name + "_in_situ")
    assert np.all(x[msk_cen] >= y[msk_cen])
    assert np.all(x[msk_sat] <= y[msk_sat])
    # Separately enforce for DBK decomposition
    for k in component_names:
        kname = name.replace("_obs", k)
        x = getattr(dbk_specphot_info, kname)
        y = getattr(dbk_specphot_info, kname + "_in_situ")
        assert np.all(x[msk_cen] >= y[msk_cen]), kname
        assert np.all(x[msk_sat] <= y[msk_sat])
        # Enforce merging is nontrivial
        assert np.any(x[msk_cen] > y[msk_cen])
        assert np.any(x[msk_sat] < y[msk_sat])

    # Enforce centrals can only get brighter and satellites can only get dimmer
    name = "obs_mags"
    x = getattr(dbk_specphot_info, name)
    y = getattr(dbk_specphot_info, name + "_in_situ")
    assert np.all(x[msk_cen] <= y[msk_cen])
    assert np.all(x[msk_sat] >= y[msk_sat])
    # Enforce merging is nontrivial
    assert np.any(x[msk_cen] < y[msk_cen])
    assert np.any(x[msk_sat] > y[msk_sat])
    # Separately enforce for DBK decomposition
    for k in component_names:
        x = getattr(dbk_specphot_info, name + k)
        y = getattr(dbk_specphot_info, name + k + "_in_situ")
        assert np.all(x[msk_cen] <= y[msk_cen])
        assert np.all(x[msk_sat] >= y[msk_sat])
        # Enforce merging is nontrivial
        assert np.any(x[msk_cen] < y[msk_cen])
        assert np.any(x[msk_sat] > y[msk_sat])

    # Enforce centrals can only get brighter lines and satellites less bright
    name = "linelum_gal"
    x = getattr(dbk_specphot_info, name)
    y = getattr(dbk_specphot_info, name + "_in_situ")
    assert np.all(x[msk_cen] >= y[msk_cen])
    assert np.all(x[msk_sat] <= y[msk_sat])
    # Separately enforce for DBK decomposition
    for k in component_names:
        kname = name.replace("_gal", k)
        x = getattr(dbk_specphot_info, kname)
        y = getattr(dbk_specphot_info, kname + "_in_situ")
        assert np.all(x[msk_cen] >= y[msk_cen])
        assert np.all(x[msk_sat] <= y[msk_sat])
        # Enforce merging is nontrivial
        assert np.any(x[msk_cen] > y[msk_cen])
        assert np.any(x[msk_sat] < y[msk_sat])
