""""""

import numpy as np
from jax import random as jran

from .. import lc_data_sfh


def _get_mc_lc_data_sfh_for_unit_testing(
    z_min=0.1,
    z_max=0.3,
    lgmp_min=13.0,
    lgmsub_min=13.0,
    sky_area_degsq=1.0,
):
    ran_key = jran.key(0)

    lc_data = lc_data_sfh.mc_lc_data_sfh(
        ran_key=ran_key,
        z_min=z_min,
        z_max=z_max,
        lgmp_min=lgmp_min,
        lgmsub_min=lgmsub_min,
        sky_area_degsq=sky_area_degsq,
    )

    return lc_data


def _get_weighted_lc_data_sfh_for_unit_testing(
    num_halos=75,
    z_min=0.1,
    z_max=3.0,
    lgmp_min=10.0,
    lgmp_max=15.0,
    sky_area_degsq=100.0,
):
    ran_key = jran.key(0)

    lc_data = lc_data_sfh.weighted_lc_data_sfh(
        ran_key=ran_key,
        n_host_halos=num_halos,
        z_min=z_min,
        z_max=z_max,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        sky_area_degsq=sky_area_degsq,
    )

    return lc_data


def test_mc_lc_data_sfh():
    lc_data = _get_mc_lc_data_sfh_for_unit_testing()
    n_tot = lc_data.z_obs.size

    shape_ntot_keys = (
        "z_obs",
        "t_obs",
        "logmp_obs",
        "logmp0",
        "t_infall",
        "logmp_infall",
        "logmhost_infall",
        "is_central",
        "halo_indx",
    )

    for lc_key in shape_ntot_keys:
        arr = getattr(lc_data, lc_key)
        assert arr.shape == (n_tot,), f"lc_data.{lc_key} has the wrong shape"
        assert np.all(np.isfinite(arr)), f"lc_data.{lc_key} has NaNs"

    for field in lc_data_sfh.LCDataSFH._fields:
        assert hasattr(lc_data, field)

    for arr in lc_data.mah_params:
        assert np.all(np.isfinite(arr))


def test_weighted_lc_data_sfh():
    num_halos = 75
    lc_data = _get_weighted_lc_data_sfh_for_unit_testing(num_halos=num_halos)
    n_tot = lc_data.z_obs.size

    shape_ntot_keys = (
        "z_obs",
        "t_obs",
        "logmp_obs",
        "logmp0",
        "t_infall",
        "logmp_infall",
        "logmhost_infall",
        "is_central",
        "halo_indx",
    )

    for lc_key in shape_ntot_keys:
        arr = getattr(lc_data, lc_key)
        assert arr.shape == (n_tot,), f"lc_data.{lc_key} has the wrong shape"
        assert np.all(np.isfinite(arr)), f"lc_data.{lc_key} has NaNs"

    for field in lc_data_sfh.LCDataSFH._fields:
        assert hasattr(lc_data, field)

    for arr in lc_data.mah_params:
        assert np.all(np.isfinite(arr))


# --- Host halo lightcone functions ---


def _get_mc_lc_data_sfh_centrals_for_unit_testing(
    z_min=0.1,
    z_max=0.3,
    lgmp_min=13.0,
    sky_area_degsq=1.0,
):
    ran_key = jran.key(0)

    lc_data = lc_data_sfh.mc_lc_data_sfh_centrals(
        ran_key=ran_key,
        z_min=z_min,
        z_max=z_max,
        lgmp_min=lgmp_min,
        sky_area_degsq=sky_area_degsq,
    )

    return lc_data


def _get_weighted_lc_data_sfh_centrals_for_unit_testing(
    num_halos=75,
    z_min=0.1,
    z_max=3.0,
    lgmp_min=10.0,
    lgmp_max=15.0,
    sky_area_degsq=100.0,
):
    ran_key = jran.key(0)

    lc_data = lc_data_sfh.weighted_lc_data_sfh_centrals(
        ran_key=ran_key,
        num_halos=num_halos,
        z_min=z_min,
        z_max=z_max,
        lgmp_min=lgmp_min,
        lgmp_max=lgmp_max,
        sky_area_degsq=sky_area_degsq,
    )

    return lc_data


def test_mc_lc_data_sfh_centrals():
    lc_data = _get_mc_lc_data_sfh_centrals_for_unit_testing()
    n_tot = lc_data.z_obs.size

    shape_ntot_keys = (
        "z_obs",
        "t_obs",
        "logmp_obs",
        "logmp0",
    )

    for lc_key in shape_ntot_keys:
        arr = getattr(lc_data, lc_key)
        assert arr.shape == (n_tot,), f"lc_data.{lc_key} has the wrong shape"
        assert np.all(np.isfinite(arr)), f"lc_data.{lc_key} has NaNs"

    for field in lc_data_sfh.LCDataSFHCentrals._fields:
        assert hasattr(lc_data, field)

    for arr in lc_data.mah_params:
        assert np.all(np.isfinite(arr))


def test_weighted_lc_data_sfh_centrals():
    num_halos = 75
    lc_data = _get_weighted_lc_data_sfh_centrals_for_unit_testing(num_halos=num_halos)
    n_tot = lc_data.z_obs.size

    shape_ntot_keys = (
        "z_obs",
        "t_obs",
        "logmp_obs",
        "logmp0",
    )

    for lc_key in shape_ntot_keys:
        arr = getattr(lc_data, lc_key)
        assert arr.shape == (n_tot,), f"lc_data.{lc_key} has the wrong shape"
        assert np.all(np.isfinite(arr)), f"lc_data.{lc_key} has NaNs"

    for field in lc_data_sfh.LCDataSFHCentrals._fields:
        assert hasattr(lc_data, field)

    for arr in lc_data.mah_params:
        assert np.all(np.isfinite(arr))
