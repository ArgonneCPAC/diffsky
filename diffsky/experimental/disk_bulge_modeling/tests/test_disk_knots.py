import numpy as np
from jax import random as jran

from dsps.sfh.diffburst import (
    DEFAULT_BURST_PARAMS,
    _pureburst_age_weights_from_params,
)
from ..disk_knots import _disk_knot_kern, _disk_knot_vmap


def test_disk_knot_kern():
    ran_key = jran.PRNGKey(0)

    n_tests = 2_000
    for __ in range(n_tests):
        keys = jran.split(ran_key, 6)
        ran_key, tobs_key, sfh_key, sfh_disk_key, burst_key, fknot_key = keys

        nt = 100
        tarr = np.linspace(0.1, 13.8, nt)
        tobs = jran.uniform(tobs_key, minval=3, maxval=tarr[-1], shape=())
        sfh = 10 ** jran.uniform(sfh_key, minval=3, maxval=2, shape=(nt,))
        sfh_disk = jran.uniform(sfh_disk_key, shape=(nt,)) * sfh
        fburst = 10 ** jran.uniform(burst_key, minval=-4, maxval=-2, shape=())
        fknot = 10 ** jran.uniform(fknot_key, minval=-4, maxval=-0.5, shape=())

        ssp_lg_age_yr = np.linspace(5, 10.5, 100)
        lgyr_peak = DEFAULT_BURST_PARAMS.lgyr_peak
        lgyr_max = DEFAULT_BURST_PARAMS.lgyr_max
        age_weights_burst = _pureburst_age_weights_from_params(ssp_lg_age_yr,
                                                               lgyr_peak, lgyr_max)
        assert np.allclose(1.0, age_weights_burst.sum(), rtol=0.01)

        lg_gyr = ssp_lg_age_yr - 9.0
        args = (tarr, tobs, sfh, sfh_disk, fburst, fknot, age_weights_burst, lg_gyr)
        _res = _disk_knot_kern(*args)
        mstar_tot, mburst, mdd, mknot, age_weights_dd, age_weights_knot = _res

        mdisk = mdd + mknot
        assert mdd < mstar_tot
        assert np.allclose(mknot, mdisk * fknot, rtol=0.01)

        assert np.allclose(1.0, age_weights_dd.sum(), rtol=0.01)
        assert np.allclose(1.0, age_weights_knot.sum(), rtol=0.01)

        ob_msk = ssp_lg_age_yr < 7
        assert age_weights_knot[ob_msk].sum() > age_weights_dd[ob_msk].sum()
        assert np.any(age_weights_knot[~ob_msk] < age_weights_dd[~ob_msk])
        assert np.any(age_weights_knot[~ob_msk] < age_weights_dd[~ob_msk])


def test_disk_knot_vmap():
    ran_key = jran.PRNGKey(0)
    keys = jran.split(ran_key, 6)
    ran_key, tobs_key, sfh_key, sfh_disk_key, burst_key, fknot_key = keys

    nt = 100
    tarr = np.linspace(0.1, 13.8, nt)

    n_gals = 10

    tobs = jran.uniform(tobs_key, minval=3, maxval=tarr[-1], shape=(n_gals,))
    sfh = 10 ** jran.uniform(sfh_key, minval=3, maxval=2, shape=(n_gals, nt))
    sfh_disk = jran.uniform(sfh_disk_key, shape=(n_gals, nt)) * sfh
    fburst = 10 ** jran.uniform(burst_key, minval=-4, maxval=-2, shape=(n_gals,))
    fknot = 10 ** jran.uniform(fknot_key, minval=-4, maxval=-0.5, shape=(n_gals,))

    n_age = 100
    ssp_lg_age_yr = np.linspace(5, 10.5, n_age)
    lgyr_peak = DEFAULT_BURST_PARAMS.lgyr_peak
    lgyr_max = DEFAULT_BURST_PARAMS.lgyr_max
    age_weights_singleburst = _pureburst_age_weights_from_params(ssp_lg_age_yr,
                                                                 lgyr_peak, lgyr_max)
    age_weights_burstpop = np.tile(age_weights_singleburst, n_gals)
    age_weights_burstpop = age_weights_burstpop.reshape((n_gals, n_age))

    lg_gyr = ssp_lg_age_yr - 9.0
    args = (tarr, tobs, sfh, sfh_disk, fburst, fknot, age_weights_burstpop, lg_gyr)
    _res = _disk_knot_vmap(*args)
    for _x in _res:
        assert np.all(np.isfinite(_x))

    mstar_tot, mburst, mdd, mknot, age_weights_dd, age_weights_knot = _res
    assert mstar_tot.shape == (n_gals,)
    assert mburst.shape == (n_gals,)
    assert mdd.shape == (n_gals,)
    assert mknot.shape == (n_gals,)
    assert age_weights_dd.shape == (n_gals, n_age)
    assert age_weights_knot.shape == (n_gals, n_age)
    assert np.all(mstar_tot > mdd + mknot)
