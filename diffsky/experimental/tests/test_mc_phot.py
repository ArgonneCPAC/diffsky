""""""

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY
from jax import random as jran

from ...param_utils import diffsky_param_wrapper as dpw
from .. import mc_diffsky_seds as mcsed
from .. import mc_phot
from . import test_lc_phot_kern as tlcphk


def test_mc_phot_kern_agrees_with_mc_diffsky_seds_phot_kern():
    """Enforce agreement to 1e-4 for the photometry computed by these two functions:
    1. mcsed._mc_diffsky_phot_kern
    2. mc_phot._mc_phot_kern

    """
    ran_key = jran.key(0)
    lc_data, tcurves = tlcphk._get_weighted_lc_data_for_unit_testing()

    fb = 0.156

    phot_info = mcsed._mc_diffsky_phot_kern(
        ran_key,
        lc_data.z_obs,
        lc_data.t_obs,
        lc_data.mah_params,
        lc_data.logmp0,
        lc_data.t_table,
        lc_data.ssp_data,
        lc_data.precomputed_ssp_mag_table,
        lc_data.z_phot_table,
        lc_data.wave_eff_table,
        *dpw.DEFAULT_PARAM_COLLECTION,
        DEFAULT_COSMOLOGY,
        fb,
    )

    phot_info2 = mc_phot._mc_phot_kern(
        ran_key,
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
    )._asdict()

    TOL = 1e-4
    for p, p2 in zip(phot_info["obs_mags"], phot_info2["obs_mags"]):
        assert np.allclose(p, p2, rtol=TOL)

    for p, p2 in zip(phot_info["dust_params"], phot_info2["dust_params"]):
        assert np.allclose(p, p2, rtol=TOL)

    assert np.allclose(phot_info["uran_av"], phot_info2["uran_av"])
    assert np.allclose(phot_info["uran_delta"], phot_info2["uran_delta"])
    assert np.allclose(phot_info["uran_funo"], phot_info2["uran_funo"])
