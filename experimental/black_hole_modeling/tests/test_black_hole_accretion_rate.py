import numpy as np
from jax import random as jran

from .. import black_hole_accretion_rate as bhar


def test_eddington_ratio_distribution():
    redshift = 0.5
    rate_table, pdf_table = bhar.eddington_ratio_distribution(redshift)
    assert np.all(np.isfinite(pdf_table))


def test_monte_carlo_eddington_ratio():
    ran_key = jran.key(0)
    npts = 5_000

    for redshift in np.arange(0, 15):
        ran_key, test_key = jran.split(ran_key, 2)
        sfr_percentile = jran.uniform(
            test_key, minval=1e-4, maxval=1 - 1e-4, shape=(npts,)
        )

        edd_ratio = bhar.monte_carlo_eddington_ratio(redshift, sfr_percentile)
        assert np.all(np.isfinite(edd_ratio))

        msk_q = sfr_percentile < 0.2
        msk_sf = sfr_percentile > 0.8
        mean_edd_ratio_q = edd_ratio[msk_q].mean()
        mean_edd_ratio_sf = edd_ratio[msk_sf].mean()
        assert mean_edd_ratio_q < mean_edd_ratio_sf

        assert np.mean(edd_ratio > 1) < 0.1
