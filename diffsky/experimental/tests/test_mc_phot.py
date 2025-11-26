""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ...param_utils import diffsky_param_wrapper as dpw
from .. import mc_diffsky_seds as mcsed
from .. import mc_phot
from . import test_lc_phot_kern as tlcphk


def test_mc_phot_kern():
    ran_key = jran.key(0)
    lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing()

    u_param_collection = dpw.get_u_param_collection_from_param_collection(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    u_param_arr = dpw.unroll_u_param_collection_into_flat_array(*u_param_collection)

    fb = 0.156
    phot_info = mcsed._mc_diffsky_phot_flat_u_params(
        u_param_arr, ran_key, lc_data, DEFAULT_COSMOLOGY, fb
    )

    phot_info2 = mc_phot._mc_phot_kern(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.t_table,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpw.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )

    TOL = 1e-4
    for p, p2 in zip(phot_info["obs_mags"], phot_info2["obs_mags"]):
        assert np.allclose(p, p2, rtol=TOL)
