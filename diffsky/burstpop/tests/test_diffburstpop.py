"""
"""
import numpy as np
from dsps.sfh import diffburst
from jax import random as jran

from .. import diffburstpop as dbp
from ..fburstpop import FburstPopUParams
from ..tburstpop import TburstPopUParams

TOL = 1e-2


def test_calc_bursty_age_weights_from_diffburstpop_params_evaluates_on_defaults():
    n_age = 107
    ssp_lg_age_gyr = np.linspace(5.5, 10.5, n_age)
    ran_key = jran.PRNGKey(0)

    n_tests = 1_000
    for __ in range(n_tests):
        ran_key, logsm_key, logssfr_key, smooth_key = jran.split(ran_key, 4)
        logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=())
        logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=())
        smooth_age_weights = jran.uniform(
            smooth_key, minval=0, maxval=1, shape=(n_age,)
        )

        smooth_age_weights = smooth_age_weights / smooth_age_weights.sum()
        args = (
            dbp.DEFAULT_DIFFBURSTPOP_PARAMS,
            logsm,
            logssfr,
            ssp_lg_age_gyr,
            smooth_age_weights,
        )
        (
            age_weights,
            burst_params,
        ) = dbp.calc_bursty_age_weights_from_diffburstpop_params(*args)
        assert age_weights.shape == (n_age,)
        assert np.all(np.isfinite(age_weights))

        unity = np.sum(age_weights)
        assert np.allclose(unity, 1.0, atol=1e-2)

        assert diffburst.LGFBURST_MIN < burst_params.lgfburst < diffburst.LGFBURST_MAX
        assert (
            diffburst.LGYR_PEAK_MIN < burst_params.lgyr_peak < diffburst.LGYR_PEAK_MAX
        )

        assert burst_params.lgyr_peak < burst_params.lgyr_max < diffburst.LGAGE_MAX


def test_calc_bursty_age_weights_from_diffburstpop_u_params_evaluates_on_u_randoms():
    n_age = 107
    ssp_lg_age_gyr = np.linspace(5.5, 10.5, n_age) - 9.0

    n_fburstpop_params = len(dbp.DEFAULT_DIFFBURSTPOP_PARAMS.fburstpop_params)
    n_tburstpop_params = len(dbp.DEFAULT_DIFFBURSTPOP_PARAMS.tburstpop_params)
    ran_key = jran.PRNGKey(0)

    n_tests = 1_000
    for __ in range(n_tests):
        ran_key, logsm_key, logssfr_key, smooth_key = jran.split(ran_key, 4)
        logsm = jran.uniform(logsm_key, minval=0, maxval=10, shape=())
        logssfr = jran.uniform(logssfr_key, minval=-12, maxval=-8, shape=())
        smooth_age_weights = jran.uniform(
            smooth_key, minval=0, maxval=1, shape=(n_age,)
        )
        ran_key, fb_key, tb_key = jran.split(ran_key, 3)
        u_fb = jran.uniform(fb_key, minval=-10, maxval=10, shape=(n_fburstpop_params,))
        u_tb = jran.uniform(tb_key, minval=-10, maxval=10, shape=(n_tburstpop_params,))
        fburstpop_u_params = FburstPopUParams(*u_fb)
        tburstpop_u_params = TburstPopUParams(*u_tb)

        diffburstpop_u_params = dbp.DiffburstPopUParams(
            fburstpop_u_params, tburstpop_u_params
        )

        smooth_age_weights = smooth_age_weights / smooth_age_weights.sum()
        args = (
            diffburstpop_u_params,
            logsm,
            logssfr,
            ssp_lg_age_gyr,
            smooth_age_weights,
        )
        (
            age_weights,
            burst_params,
        ) = dbp.calc_bursty_age_weights_from_diffburstpop_u_params(*args)
        assert age_weights.shape == (n_age,)
        assert np.all(np.isfinite(age_weights))

        unity = np.sum(age_weights)
        assert np.allclose(unity, 1.0, atol=1e-2)

        assert diffburst.LGFBURST_MIN <= burst_params.lgfburst <= diffburst.LGFBURST_MAX
        assert (
            diffburst.LGYR_PEAK_MIN <= burst_params.lgyr_peak <= diffburst.LGYR_PEAK_MAX
        )

        assert burst_params.lgyr_peak <= burst_params.lgyr_max <= diffburst.LGAGE_MAX


def test_diffburstpop_u_param_inversion():
    u_params = dbp.get_unbounded_diffburstpop_params(dbp.DEFAULT_DIFFBURSTPOP_PARAMS)
    u_params_fburst, u_params_tburst = u_params

    assert np.allclose(
        dbp.DEFAULT_DIFFBURSTPOP_U_PARAMS.fburstpop_u_params, u_params_fburst, rtol=TOL
    )

    assert np.allclose(
        dbp.DEFAULT_DIFFBURSTPOP_U_PARAMS.tburstpop_u_params, u_params_tburst, rtol=TOL
    )
