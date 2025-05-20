""" """

from diffstarpop import get_unbounded_diffstarpop_params
from dsps.metallicity import get_unbounded_mzr_params
from jax import jit as jjit

from diffsky.dustpop import tw_dustpop_mono_noise as twdp
from diffsky.ssp_err_model import ssp_err_model


def get_unbounded_spspop_params_tw_dust():
    raise NotImplementedError()


@jjit
def get_u_param_collection_from_param_collection(
    diffstarpop_params,
    mzr_params,
    spspop_params,
    dustpop_scatter_params,
    ssp_err_pop_params,
):
    diffstarpop_u_params = get_unbounded_diffstarpop_params(diffstarpop_params)
    mzr_u_params = get_unbounded_mzr_params(mzr_params)
    spspop_u_params = get_unbounded_spspop_params_tw_dust(spspop_params)
    dustpop_scatter_u_params = twdp.get_unbounded_dustpop_scatter_params(
        dustpop_scatter_params
    )
    ssp_err_pop_u_params = ssp_err_model.get_unbounded_ssperr_params(ssp_err_pop_params)

    u_param_collection = (
        diffstarpop_u_params,
        mzr_u_params,
        spspop_u_params,
        dustpop_scatter_u_params,
        ssp_err_pop_u_params,
    )
    return u_param_collection


@jjit
def get_param_tuple_from_u_param_array():
    pass
