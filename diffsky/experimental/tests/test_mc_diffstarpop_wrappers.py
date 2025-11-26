""""""

import numpy as np
from diffstar.defaults import FB, T_TABLE_MIN
from diffstar.diffstarpop.defaults import DEFAULT_DIFFSTARPOP_PARAMS
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.data_loaders.defaults import TransmissionCurve
from dsps.metallicity import umzr
from jax import random as jran

from .. import lc_phot_kern
from .. import mc_diffstarpop_wrappers as mcdw
from .. import mc_lightcone_halos as mclh


def test_diffstarpop_lc_cen_wrapper_agrees_with_lc_phot_kern():

    ran_key = jran.key(0)
    lgmp_min = 12.0
    z_min, z_max = 0.1, 0.5
    sky_area_degsq = 1.0

    args = (ran_key, lgmp_min, z_min, z_max, sky_area_degsq)

    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(*args)
    t0 = flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    t_table = np.linspace(T_TABLE_MIN, t0, 100)

    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        ran_key,
        lc_halopop["mah_params"],
        lc_halopop["logmp0"],
        t_table,
        lc_halopop["t_obs"],
        DEFAULT_COSMOLOGY,
        FB,
    )
    diffstar_galpop = lc_phot_kern.diffstarpop_lc_cen_wrapper(*args)
    diffstar_galpop2 = mcdw.diffstarpop_lc_cen_wrapper(*args)

    assert np.allclose(diffstar_galpop.frac_q, diffstar_galpop2.frac_q)
    assert np.allclose(diffstar_galpop.sfh_ms, diffstar_galpop2.sfh_ms)
    assert np.allclose(diffstar_galpop.sfh_q, diffstar_galpop2.sfh_q)
    for p, p2 in zip(
        diffstar_galpop.diffstar_params_ms, diffstar_galpop2.diffstar_params_ms
    ):
        assert np.allclose(p, p2)
    for p, p2 in zip(
        diffstar_galpop.diffstar_params_q, diffstar_galpop2.diffstar_params_q
    ):
        assert np.allclose(p, p2)
