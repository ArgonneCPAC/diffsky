""" """

import numpy as np
from jax import random as jran

from ...param_utils import diffsky_param_wrapper as dpw
from .. import lc_phot_kern
from .. import mc_diffsky_seds as mcsed
from . import test_lc_phot_kern as tlcphk


def test_mc_diffsky_seds():
    ran_key = jran.key(0)
    lc_data = lc_data = tlcphk._generate_sobol_lc_data()

    n_z_table, n_bands, n_met, n_age = lc_data.precomputed_ssp_mag_table.shape
    assert lc_data.z_phot_table.shape == (n_z_table,)
    assert lc_data.ssp_data.ssp_lgmet.shape == (n_met,)
    assert lc_data.ssp_data.ssp_lg_age_gyr.shape == (n_age,)

    u_param_collection = dpw.get_u_param_collection_from_param_collection(
        *dpw.DEFAULT_PARAM_COLLECTION
    )
    u_param_arr = dpw.unroll_u_param_collection_into_flat_array(*u_param_collection)

    lc_phot = mcsed.mc_diffsky_seds(u_param_arr, ran_key, lc_data)
    lc_phot_orig = lc_phot_kern.multiband_lc_phot_kern_u_param_arr(
        u_param_arr, ran_key, lc_data
    )
    for pname, pval in zip(lc_phot_orig._fields, lc_phot_orig):
        pval2 = getattr(lc_phot, pname)
        assert np.allclose(pval, pval2)
