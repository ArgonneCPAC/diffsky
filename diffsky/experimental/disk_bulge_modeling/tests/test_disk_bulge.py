import numpy as np
from diffstar.utils import cumulative_mstar_formed_galpop
from jax import random as jran

from ...disk_bulge_modeling.disk_knots import FKNOT_MAX
from ...disk_bulge_modeling.mc_disk_bulge import (
    mc_disk_bulge,
)
from dsps.sfh.diffburst import (
    DEFAULT_BURST_PARAMS,
)
from dsps.sfh.diffburst import (
    _pureburst_age_weights_from_params as _burst_age_weights_from_params,
)
from ..disk_bulge_kernels import (
    DEFAULT_FBULGE_PARAMS,
    FBULGE_MAX,
    FBULGE_MIN,
    _bulge_fraction_vs_tform,
    _bulge_sfh,
    _burst_age_weights_from_params_vmap,
    _decompose_sfh_singlegal_into_bulge_disk_knots,
    _decompose_sfhpop_into_bulge_disk_knots,
    _linterp_vmap,
    calc_tform_kern,
    decompose_sfhpop_into_bulge_disk_knots,
)

DEFAULT_T10, DEFAULT_T90 = 2.0, 9.0


def test_bulge_sfh():
    nt = 100
    tarr = np.linspace(0.1, 13.8, nt)
    sfh = np.ones_like(tarr)
    _res = _bulge_sfh(tarr, sfh, DEFAULT_FBULGE_PARAMS)
    for x in _res:
        assert np.all(np.isfinite(x))
        assert x.shape == (nt,)
    smh, fbulge, sfh_bulge, smh_bulge, bth = _res


def test_bulge_fraction_vs_tform():
    nt = 100
    tarr = np.linspace(0.1, 13.8, nt)
    t10, t90 = 2.0, 10.0
    fbulge = _bulge_fraction_vs_tform(tarr, t10, t90, DEFAULT_FBULGE_PARAMS)
    assert np.all(np.isfinite(fbulge))
    assert np.all(fbulge > FBULGE_MIN)
    assert np.all(fbulge < FBULGE_MAX)


def test_calc_tform_kern():
    tarr = np.linspace(0.1, 13.8, 200)
    smh = np.logspace(5, 12, tarr.size)
    t10 = calc_tform_kern(tarr, smh, 0.1)
    t90 = calc_tform_kern(tarr, smh, 0.9)
    assert t10 < t90


def test_decompose_sfh_into_bulge_disk_knots():
    """Enforce physically reasonable disk/bulge/knot decomposition for some random
    galaxy distributions. Run n_tests tests for galaxy populations with
    different distributions of {Fburst, t_obs}

    """
    ran_key = jran.PRNGKey(0)

    n_age = 40
    ssp_lg_age_yr = np.linspace(5, 10.25, n_age)
    ssp_lg_age_gyr = ssp_lg_age_yr - 9.0

    n_t = 100
    t0 = 13.8
    t_table_min = 0.01
    gal_t_table = np.linspace(t_table_min, t0, n_t)

    n_gals = 1_000
    n_tests = 20
    for itest in range(n_tests):
        itest_key, ran_key = jran.split(ran_key, 2)

        # make sure t_obs > t_table_min
        t_obs_min = 0.2
        gal_t_obs = np.random.uniform(t_obs_min, t0 - 0.05, n_gals)
        sfh_peak_max = np.random.uniform(0, 200)
        sfh_peak = np.random.uniform(0, sfh_peak_max, n_gals)
        gal_sfh_u = np.random.uniform(0, 1, n_gals * n_t).reshape((n_gals, n_t))
        gal_sfh = gal_sfh_u * sfh_peak.reshape((n_gals, 1))

        LGFBURST_UPPER_BOUND = -1
        lgfb_max = np.random.uniform(-3, LGFBURST_UPPER_BOUND)
        lgfb_min = np.random.uniform(lgfb_max - 3, lgfb_max)
        gal_fburst = 10 ** np.random.uniform(lgfb_min, lgfb_max, n_gals)

        gal_burstshape_params = np.tile(DEFAULT_BURST_PARAMS, n_gals)
        gal_burstshape_params = gal_burstshape_params.reshape((n_gals, 3))

        gal_burst_age_weights = _burst_age_weights_from_params_vmap(
            ssp_lg_age_yr, gal_burstshape_params[:, 1], gal_burstshape_params[:, 2],
        )

        age_weights_singleburst = _burst_age_weights_from_params(
            ssp_lg_age_yr, DEFAULT_BURST_PARAMS.lgyr_peak,
            DEFAULT_BURST_PARAMS.lgyr_max,
        )
        age_weights_burstpop = np.tile(age_weights_singleburst, n_gals)
        age_weights_burstpop = age_weights_burstpop.reshape((n_gals, n_age))
        assert np.allclose(gal_burst_age_weights, age_weights_burstpop, rtol=0.001)

        gal_fknot = np.random.uniform(0, FKNOT_MAX, n_gals)
        gal_fbulge_params = mc_disk_bulge(itest_key, gal_t_table, gal_sfh)[0]

        args = (
            gal_fbulge_params,
            gal_fknot,
            gal_t_obs,
            gal_t_table,
            gal_sfh,
            gal_fburst,
            age_weights_burstpop,
            ssp_lg_age_gyr,
        )
        _res = _decompose_sfhpop_into_bulge_disk_knots(*args)
        _res2 = decompose_sfhpop_into_bulge_disk_knots(
            gal_fbulge_params,
            gal_fknot,
            gal_t_obs,
            gal_t_table,
            gal_sfh,
            gal_fburst,
            gal_burstshape_params,
            ssp_lg_age_gyr,
        )
        # Enforce no NaN and that the convenience function agrees with the kernel
        for x, x2 in zip(_res, _res2):
            assert np.all(np.isfinite(x))
            assert np.allclose(x, x2, rtol=1e-4)

        mbulge, mdd, mknot, mburst = _res[:4]
        bulge_age_weights, dd_age_weights, knot_age_weights = _res[4:7]
        bulge_sfh, frac_bulge_t_obs = _res[7:]

        assert mbulge.shape == (n_gals,)
        assert mdd.shape == (n_gals,)
        assert mknot.shape == (n_gals,)
        assert mburst.shape == (n_gals,)
        assert bulge_age_weights.shape == (n_gals, n_age)
        assert dd_age_weights.shape == (n_gals, n_age)
        assert knot_age_weights.shape == (n_gals, n_age)
        assert bulge_sfh.shape == (n_gals, n_t)
        assert frac_bulge_t_obs.shape == (n_gals,)

        # Each galaxy's age weights should be a unit-normalized PDF
        assert np.allclose(np.sum(bulge_age_weights, axis=1), 1.0, rtol=0.01)
        assert np.allclose(np.sum(dd_age_weights, axis=1), 1.0, rtol=0.01)
        assert np.allclose(np.sum(knot_age_weights, axis=1), 1.0, rtol=0.01)

        # The sum of the masses in each component should equal
        # # the total stellar mass formed at gal_t_obs
        mtot = mbulge + mdd + mknot
        lgmtot = np.log10(mtot)

        gal_smh = cumulative_mstar_formed_galpop(gal_t_table, gal_sfh)
        gal_logsmh = np.log10(gal_smh)

        lgt_table = np.log10(gal_t_table)
        lgmstar_t_obs = _linterp_vmap(np.log10(gal_t_obs), lgt_table, gal_logsmh)

        assert np.allclose(lgmstar_t_obs, lgmtot, atol=0.01)

        # Star-forming knots should never have fewer young stars than the diffuse disk
        dd_age_cdf = np.cumsum(dd_age_weights, axis=1)
        knot_age_cdf = np.cumsum(knot_age_weights, axis=1)

        tol = 0.01
        indx_ostar = np.searchsorted(ssp_lg_age_yr, 6.5)
        assert np.all(dd_age_cdf[:, indx_ostar] <= knot_age_cdf[:, indx_ostar] + tol)
        assert np.any(dd_age_cdf[:, indx_ostar] < knot_age_cdf[:, indx_ostar])

        # The bulge should never have more young stars than star-forming knots
        tol = 0.01
        bulge_age_cdf = np.cumsum(bulge_age_weights, axis=1)
        assert np.all(bulge_age_cdf[:, indx_ostar] <= knot_age_cdf[:, indx_ostar] + tol)

        # On average, star-forming knots should be younger than diffuse disks
        # and diffuse disks should should be younger than bulges
        bulge_median_frac_ob_stars = np.median(bulge_age_cdf[:, indx_ostar])
        dd_median_frac_ob_stars = np.median(dd_age_cdf[:, indx_ostar])
        knot_median_frac_ob_stars = np.median(knot_age_cdf[:, indx_ostar])
        assert (
            bulge_median_frac_ob_stars
            < dd_median_frac_ob_stars
            < knot_median_frac_ob_stars
        )

        # Bulge SFH should never exceed total SFH
        assert np.all(bulge_sfh <= gal_sfh)

        # Bulge SFH should fall below total SFH at some point in history of each galaxy
        assert np.all(np.any(bulge_sfh < gal_sfh, axis=1))

        # Bulge fraction should respect 0 < frac_bulge < 1 for every galaxy
        assert np.all(frac_bulge_t_obs > 0)
        assert np.all(frac_bulge_t_obs < 1)
        # Bulge fraction should respect 0 < frac_bulge < 1 for every galaxy
        assert np.all(frac_bulge_t_obs > 0)
        assert np.all(frac_bulge_t_obs < 1)
        assert np.all(frac_bulge_t_obs < 1)
        assert np.all(frac_bulge_t_obs < 1)


def test_decompose_sfh_singlegal_into_bulge_disk_knots_agrees_with_vmap():
    ran_key = jran.PRNGKey(0)

    n_age = 40
    ssp_lg_age_yr = np.linspace(5, 10.25, n_age)
    ssp_lg_age_gyr = ssp_lg_age_yr - 9.0

    n_t = 100
    t0 = 13.8
    t_table_min = 0.01
    gal_t_table = np.linspace(t_table_min, t0, n_t)

    n_gals = 10

    itest_key, ran_key = jran.split(ran_key, 2)

    # make sure t_obs > t_table_min
    t_obs_min = 0.2
    gal_t_obs = np.random.uniform(t_obs_min, t0 - 0.05, n_gals)
    sfh_peak_max = np.random.uniform(0, 200)
    sfh_peak = np.random.uniform(0, sfh_peak_max, n_gals)
    gal_sfh_u = np.random.uniform(0, 1, n_gals * n_t).reshape((n_gals, n_t))
    gal_sfh = gal_sfh_u * sfh_peak.reshape((n_gals, 1))

    LGFBURST_UPPER_BOUND = -1
    lgfb_max = np.random.uniform(-3, LGFBURST_UPPER_BOUND)
    lgfb_min = np.random.uniform(lgfb_max - 3, lgfb_max)
    gal_fburst = 10 ** np.random.uniform(lgfb_min, lgfb_max, n_gals)

    gal_burstshape_params = np.tile(DEFAULT_BURST_PARAMS, n_gals)
    gal_burstshape_params = gal_burstshape_params.reshape((n_gals, 3))

    gal_burst_age_weights = _burst_age_weights_from_params_vmap(
        ssp_lg_age_yr, gal_burstshape_params[:, 1], gal_burstshape_params[:, 2],
    )
    age_weights_singleburst = _burst_age_weights_from_params(
        ssp_lg_age_yr, DEFAULT_BURST_PARAMS.lgyr_peak,
        DEFAULT_BURST_PARAMS.lgyr_max,
    )

    age_weights_burstpop = np.tile(age_weights_singleburst, n_gals)
    age_weights_burstpop = age_weights_burstpop.reshape((n_gals, n_age))
    assert np.allclose(gal_burst_age_weights, age_weights_burstpop, rtol=0.001)

    gal_fknot = np.random.uniform(0, FKNOT_MAX, n_gals)
    gal_fbulge_params = mc_disk_bulge(itest_key, gal_t_table, gal_sfh)[0]

    args = (
        gal_fbulge_params,
        gal_fknot,
        gal_t_obs,
        gal_t_table,
        gal_sfh,
        gal_fburst,
        age_weights_burstpop,
        ssp_lg_age_gyr,
    )
    _res_vmap = _decompose_sfhpop_into_bulge_disk_knots(*args)

    for igal in range(n_gals):
        fbulge_params = gal_fbulge_params[igal, :]
        fknot = gal_fknot[igal]
        t_obs = gal_t_obs[igal]
        sfh_table = gal_sfh[igal, :]
        fburst = gal_fburst[igal]
        age_weights_burst = age_weights_burstpop[igal, :]
        args = (
            fbulge_params,
            fknot,
            t_obs,
            gal_t_table,
            sfh_table,
            fburst,
            age_weights_burst,
            ssp_lg_age_gyr,
        )
        _res_igal = _decompose_sfh_singlegal_into_bulge_disk_knots(*args)
        for x_vmap, x_igal in zip(_res_vmap, _res_igal):
            assert np.allclose(x_vmap[igal], x_igal)
