""" """

import numpy as np
from dsps.cosmology import flat_wcdm
from jax import random as jran

from diffsky.mass_functions import mc_hosts

from .. import mc_lightcone_halos as mclh


def test_mc_lightcone_host_halos():
    """Enforce mc_lightcone_host_halos produces consistent halo mass functions as
    the diffsky.mass_functions.mc_hosts function evaluated at the median redshift

    """
    lgmp_min = 12.0
    z_min, z_max = 0.1, 0.5
    sky_area_degsq = 10.0

    n_tests = 5
    ran_keys = jran.split(jran.key(0), n_tests)
    for ran_key in ran_keys:
        args = (ran_key, lgmp_min, z_min, z_max, sky_area_degsq)

        redshifts_galpop, logmp_halopop = mclh.mc_lightcone_host_halos(*args)
        assert np.all(np.isfinite(redshifts_galpop))
        assert np.all(np.isfinite(logmp_halopop))
        assert logmp_halopop.shape == redshifts_galpop.shape
        assert np.all(redshifts_galpop >= z_min)
        assert np.all(redshifts_galpop <= z_max)
        assert np.all(logmp_halopop > lgmp_min)

        z_med = np.median(redshifts_galpop)

        cosmo_params = flat_wcdm.PLANCK15
        vol_lo = (
            (4 / 3)
            * np.pi
            * flat_wcdm.comoving_distance_to_z(z_min, *cosmo_params) ** 3
        )
        vol_hi = (
            (4 / 3)
            * np.pi
            * flat_wcdm.comoving_distance_to_z(z_max, *cosmo_params) ** 3
        )
        fsky = sky_area_degsq / mclh.FULL_SKY_AREA
        vol_com = fsky * (vol_hi - vol_lo)

        lgmp_halopop_zmed = mc_hosts.mc_host_halos_singlez(
            ran_key, lgmp_min, z_med, vol_com
        )

        n_lightcone, n_snapshot = redshifts_galpop.size, lgmp_halopop_zmed.size
        fracdiff = (n_lightcone - n_snapshot) / n_snapshot
        assert np.abs(fracdiff) < 0.05

        lgmp_hist_lc, lgmp_bins = np.histogram(logmp_halopop, bins=50)
        lgmp_hist_zmed, lgmp_bins = np.histogram(lgmp_halopop_zmed, bins=lgmp_bins)
        msk_counts = lgmp_hist_zmed > 500
        fracdiff = (
            lgmp_hist_lc[msk_counts] - lgmp_hist_zmed[msk_counts]
        ) / lgmp_hist_zmed[msk_counts]
        assert np.all(np.abs(fracdiff) < 0.1)
