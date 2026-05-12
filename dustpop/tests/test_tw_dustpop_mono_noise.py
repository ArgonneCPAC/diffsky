""" """

import numpy as np

from .. import tw_dustpop_mono as twd
from .. import tw_dustpop_mono_noise as twdn

from ...experimental.scatter import DEFAULT_SCATTER_PARAMS

TOL = 1e-4


def test_calc_ftrans_singlegal_singlewave_from_dustpop_params():
    wave_aa = 5_000.0
    logsm = 10.0
    logssfr = -10.0
    redshift = 0.5

    n_age = 37
    ssp_lg_age_gyr = np.linspace(5, 10.1, n_age) - 9.0

    random_draw_av = 0.5
    random_draw_delta = 0.5
    random_draw_funo = 0.5
    args = (
        twd.DEFAULT_DUSTPOP_PARAMS,
        wave_aa,
        logsm,
        logssfr,
        redshift,
        ssp_lg_age_gyr,
        random_draw_av,
        random_draw_delta,
        random_draw_funo,
        DEFAULT_SCATTER_PARAMS,
    )
    _res = twdn.calc_ftrans_singlegal_singlewave_from_dustpop_params(*args)
    ftrans, noisy_ftrans, dust_params, noisy_dust_params = _res
    assert np.all(np.isfinite(ftrans))
    assert ftrans.shape == (n_age,)
    assert np.all(ftrans >= 0)
    assert np.all(ftrans <= 1)
