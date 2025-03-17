""" """

import numpy as np

from .. import tw_dustpop_mono as twd
from ..avpop_mono import DEFAULT_AVPOP_U_PARAMS
from ..deltapop import DEFAULT_DELTAPOP_U_PARAMS
from ..funopop_ssfr import DEFAULT_FUNOPOP_U_PARAMS

TOL = 1e-2


def test_dustpop_u_param_inversion():
    u_params = twd.get_unbounded_dustpop_params(twd.DEFAULT_DUSTPOP_PARAMS)
    avpop_u_params, deltapop_u_params, funopop_u_params = u_params
    assert np.allclose(DEFAULT_AVPOP_U_PARAMS, avpop_u_params, rtol=TOL)
    assert np.allclose(DEFAULT_DELTAPOP_U_PARAMS, deltapop_u_params, rtol=TOL)
    assert np.allclose(DEFAULT_FUNOPOP_U_PARAMS, funopop_u_params, rtol=TOL)


def test_calc_dust_ftrans_galpop_from_dustpop_params():
    n_age = 94
    ssp_lg_age_gyr = np.linspace(5.5, 10.25, n_age) - 9.0
    logsm, redshift, logssfr = 10.0, 2.0, -10.5
    wave_aa = 500.0
    n_gals = 20
    zz = np.zeros(n_gals)
    args = (
        twd.DEFAULT_DUSTPOP_PARAMS,
        wave_aa,
        logsm + zz,
        logssfr + zz,
        redshift + zz,
        ssp_lg_age_gyr,
    )
    ftrans, dust_params = twd.calc_dust_ftrans_galpop_from_dustpop_params(*args)
    assert ftrans.shape == (n_gals, n_age)
    assert np.all(np.isfinite(ftrans))
    assert np.all(ftrans >= 0)
    assert np.all(ftrans <= 1)
    assert np.any(ftrans > 0)
    assert np.any(ftrans < 1)

    delta_age_ftrans = np.diff(ftrans, axis=1)
    assert np.all(delta_age_ftrans >= 0)
    assert np.any(delta_age_ftrans > 0)

    args = (
        twd.DEFAULT_DUSTPOP_PARAMS.avpop_params,
        logsm + zz,
        logssfr + zz,
        redshift + zz,
        ssp_lg_age_gyr,
    )
    res2 = twd.get_av_from_avpop_params_galpop(*args)
    assert res2.shape == (n_gals, n_age)

    delta = twd.get_delta_from_deltapop_params(
        twd.DEFAULT_DUSTPOP_PARAMS.deltapop_params, logsm, logssfr
    )
    assert delta.shape == ()
    assert np.all(np.isfinite(delta))

    delta = twd.get_delta_from_deltapop_params(
        twd.DEFAULT_DUSTPOP_PARAMS.deltapop_params, logsm + zz, logssfr + zz
    )
    assert delta.shape == (n_gals,)
    assert np.all(np.isfinite(delta))

    funo = twd.get_funo_from_funopop_params(
        twd.DEFAULT_DUSTPOP_PARAMS.funopop_params, logssfr
    )
    assert funo.shape == ()
    assert np.all(np.isfinite(funo))
    assert np.all(funo >= 0)
    assert np.all(funo <= 1)

    funo = twd.get_funo_from_funopop_params(
        twd.DEFAULT_DUSTPOP_PARAMS.funopop_params, logssfr + zz
    )
    assert funo.shape == (n_gals,)
    assert np.all(np.isfinite(funo))
    assert np.all(funo >= 0)
    assert np.all(funo <= 1)


def test_calc_dust_ftrans_scalar_from_default_dustpop_params_behaves_as_expected():
    ssp_lg_age_gyr = -9.0
    logsm, redshift, logssfr = 10.0, 2.0, -10.5

    # UV wavelength
    wave_aa = 400.0

    args = (
        twd.DEFAULT_DUSTPOP_PARAMS,
        wave_aa,
        logsm,
        logssfr,
        redshift,
        ssp_lg_age_gyr,
    )
    ftrans_uv, dust_params = twd.calc_dust_ftrans_scalar_from_dustpop_params(*args)
    assert np.all(np.isfinite(ftrans_uv))
    assert ftrans_uv.shape == ()
    assert np.all(ftrans_uv >= 0)
    assert np.all(ftrans_uv <= 1)
    assert np.any(ftrans_uv > 0)
    assert np.any(ftrans_uv < 1)

    # NIR wavelength
    wave_nm = 900
    wave_aa = wave_nm * 10

    args = (
        twd.DEFAULT_DUSTPOP_PARAMS,
        wave_aa,
        logsm,
        logssfr,
        redshift,
        ssp_lg_age_gyr,
    )
    ftrans_nir, dust_params = twd.calc_dust_ftrans_scalar_from_dustpop_params(*args)
    assert ftrans_nir > ftrans_uv

    assert np.all(np.isfinite(ftrans_nir))
    assert ftrans_nir.shape == ()
    assert np.all(ftrans_nir >= 0)
    assert np.all(ftrans_nir <= 1)
    assert np.any(ftrans_nir > 0)
    assert np.any(ftrans_nir < 1)

    # Mid-IR wavelength
    wave_nm = 2_000
    wave_aa = wave_nm * 10

    args = (
        twd.DEFAULT_DUSTPOP_PARAMS,
        wave_aa,
        logsm,
        logssfr,
        redshift,
        ssp_lg_age_gyr,
    )
    ftrans_fir, dust_params = twd.calc_dust_ftrans_scalar_from_dustpop_params(*args)
    assert ftrans_fir > ftrans_nir
    assert ftrans_fir > 0.9
