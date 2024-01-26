"""
"""
import numpy as np

from ..sbl18_photgrad import (
    DEFAULT_SPSPOP_PARAMS,
    DEFAULT_SPSPOP_U_PARAMS,
    get_bounded_spspop_params,
    get_unbounded_spspop_params,
)


def test_sbl18pop_param_bounding():
    params = get_bounded_spspop_params(DEFAULT_SPSPOP_U_PARAMS)
    u_params = get_unbounded_spspop_params(DEFAULT_SPSPOP_PARAMS)

    for x, y in zip(params.burstpop_params, DEFAULT_SPSPOP_PARAMS.burstpop_params):
        assert np.allclose(x, y)

    for x, y in zip(params.dustpop_params, DEFAULT_SPSPOP_PARAMS.dustpop_params):
        assert np.allclose(x, y)

    for x, y in zip(
        u_params.u_burstpop_params, DEFAULT_SPSPOP_U_PARAMS.u_burstpop_params
    ):
        assert np.allclose(x, y)

    for x, y in zip(
        u_params.u_dustpop_params, DEFAULT_SPSPOP_U_PARAMS.u_dustpop_params
    ):
        assert np.allclose(x, y)
