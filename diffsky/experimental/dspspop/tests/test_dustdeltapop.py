"""
"""
import numpy as np

from ..boris_dust import DEFAULT_U_PARAMS as DEFAULT_FUNO_U_PARAMS
from ..dust_deltapop import (
    DEFAULT_DUST_DELTA_PARAMS,
    DEFAULT_DUST_DELTA_U_PARAMS,
    _get_bounded_dust_delta_params,
    _get_dust_delta_galpop_from_params,
    _get_dust_delta_galpop_from_u_params,
    _get_unbounded_dust_delta_params,
)
from ..dustpop import (
    _frac_dust_transmission_lightcone_kernel,
    _frac_dust_transmission_singlez_kernel,
)
from ..lgavpop import DEFAULT_LGAV_U_PARAMS


def test_get_bursty_age_weights_pop_evaluates():
    n_gals = 500
    gal_logsm = np.random.uniform(0, 10, size=(n_gals,))
    gal_logssfr = np.random.uniform(-12, -8, size=(n_gals,))

    gal_dust_delta = _get_dust_delta_galpop_from_params(
        gal_logsm, gal_logssfr, DEFAULT_DUST_DELTA_PARAMS
    )
    assert gal_dust_delta.shape == (n_gals,)


def test_get_bursty_age_weights_pop_u_param_inversion():
    assert np.allclose(
        DEFAULT_DUST_DELTA_PARAMS,
        _get_bounded_dust_delta_params(DEFAULT_DUST_DELTA_U_PARAMS),
        rtol=1e-3,
    )

    inferred_default_params = _get_bounded_dust_delta_params(
        _get_unbounded_dust_delta_params(DEFAULT_DUST_DELTA_PARAMS)
    )
    assert np.allclose(DEFAULT_DUST_DELTA_PARAMS, inferred_default_params, rtol=1e-3)

    n_gals = 500
    gal_logsm = np.random.uniform(0, 10, size=(n_gals,))
    gal_logssfr = np.random.uniform(-12, -8, size=(n_gals,))

    gal_dust_delta = _get_dust_delta_galpop_from_params(
        gal_logsm, gal_logssfr, DEFAULT_DUST_DELTA_PARAMS
    )
    assert gal_dust_delta.shape == (n_gals,)
    assert np.all(np.isfinite(gal_dust_delta))

    gal_dust_delta_u = _get_dust_delta_galpop_from_u_params(
        gal_logsm, gal_logssfr, DEFAULT_DUST_DELTA_U_PARAMS
    )
    assert gal_dust_delta_u.shape == (n_gals,)
    assert np.all(np.isfinite(gal_dust_delta_u))

    assert np.allclose(gal_dust_delta, gal_dust_delta_u, rtol=1e-4)


def test_compute_dust_transmission_fractions():
    att_curve_key = 0
    ngals = 20
    z_obs = 0.1
    gal_z_obs = np.zeros(ngals) + z_obs
    logsm = np.linspace(8, 12, ngals)
    logssfr = np.linspace(-13, -8, ngals)
    gal_logfburst = np.zeros(ngals) - 3.0

    n_age = 50
    ssp_lg_age_gyr = np.linspace(5, 10.5, n_age) - 9

    n_filters, n_wave = 5, 1000
    filter_waves = np.tile(np.linspace(100, 2_000, n_wave), n_filters).reshape(
        (n_filters, n_wave)
    )
    filter_trans = np.ones((n_filters, n_wave))

    args = (
        att_curve_key,
        gal_z_obs,
        logsm,
        logssfr,
        gal_logfburst,
        ssp_lg_age_gyr,
        filter_waves,
        filter_trans,
        DEFAULT_LGAV_U_PARAMS,
        DEFAULT_DUST_DELTA_U_PARAMS,
        DEFAULT_FUNO_U_PARAMS,
    )
    _res = _frac_dust_transmission_lightcone_kernel(*args)
    gal_frac_trans, gal_att_curve_params, gal_frac_unobs = _res

    assert gal_frac_trans.shape == (ngals, n_age, n_filters)
    assert gal_att_curve_params.shape == (ngals, 3)
    assert gal_frac_unobs.shape == (ngals, n_age)

    args2 = (
        att_curve_key,
        z_obs,
        logsm,
        logssfr,
        gal_logfburst,
        ssp_lg_age_gyr,
        filter_waves,
        filter_trans,
        DEFAULT_LGAV_U_PARAMS,
        DEFAULT_DUST_DELTA_U_PARAMS,
        DEFAULT_FUNO_U_PARAMS,
    )
    gal_frac_trans2 = _frac_dust_transmission_singlez_kernel(*args2)[0]
    assert np.allclose(gal_frac_trans, gal_frac_trans2)
