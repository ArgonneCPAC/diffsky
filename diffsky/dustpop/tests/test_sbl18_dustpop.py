"""
"""
import numpy as np

from .. import sbl18_dustpop as twd
from ..avpop import DEFAULT_AVPOP_U_PARAMS
from ..deltapop import DEFAULT_DELTAPOP_U_PARAMS
from ..funopop import DEFAULT_FUNOPOP_U_PARAMS

TOL = 1e-2


def test_get_funo_from_u_params_singlegal():
    n_age = 94
    ssp_lg_age_gyr = np.linspace(5.5, 10.25, n_age) - 9.0
    logsm, logfburst, logssfr = 10.0, -2.0, -10.5

    wave_aa = 500.0

    args = (
        twd.DEFAULT_DUSTPOP_PARAMS,
        wave_aa,
        logsm,
        logssfr,
        logfburst,
        ssp_lg_age_gyr,
    )
    res = twd.calc_dust_ftrans_singlegal_singlewave_from_dustpop_params(*args)
    assert np.all(np.isfinite(res))


def test_dustpop_u_param_inversion():
    u_params = twd.get_unbounded_dustpop_params(twd.DEFAULT_DUSTPOP_PARAMS)
    avpop_u_params, deltapop_u_params, funopop_u_params = u_params
    assert np.allclose(DEFAULT_AVPOP_U_PARAMS, avpop_u_params, rtol=TOL)
    assert np.allclose(DEFAULT_DELTAPOP_U_PARAMS, deltapop_u_params, rtol=TOL)
    assert np.allclose(DEFAULT_FUNOPOP_U_PARAMS, funopop_u_params, rtol=TOL)
