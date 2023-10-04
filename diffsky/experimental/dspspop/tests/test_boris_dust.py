"""
"""
import numpy as np

from ..boris_dust import DEFAULT_U_PARAMS as DEFAULT_BORIS_U_PARAMS
from ..boris_dust import (
    _get_funo_from_params_singlegal,
    _get_funo_from_u_params_singlegal,
    get_params_from_u_params,
    get_u_params_from_params,
)


def test_param_u_param_inversion():
    p = get_params_from_u_params(DEFAULT_BORIS_U_PARAMS)
    u_p = get_u_params_from_params(p)
    assert np.allclose(u_p, DEFAULT_BORIS_U_PARAMS, rtol=1e-3)


def test_get_funo_from_u_params_singlegal():
    logsm, logfburst, logssfr = 10.0, -2.0, -10.5

    n_age = 94
    ssp_lg_age_gyr = np.linspace(5.5, 10.25, n_age) - 9.0
    funo = _get_funo_from_u_params_singlegal(
        logsm, logfburst, logssfr, ssp_lg_age_gyr, DEFAULT_BORIS_U_PARAMS
    )
    assert funo.shape == (n_age,)
    assert np.all(funo >= 0)
    assert np.all(funo <= 1)

    params = get_params_from_u_params(DEFAULT_BORIS_U_PARAMS)
    funo2 = _get_funo_from_params_singlegal(
        logsm, logfburst, logssfr, ssp_lg_age_gyr, params
    )
    assert np.allclose(funo, funo2, rtol=1e-3)
