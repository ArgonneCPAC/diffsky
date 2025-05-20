""" """

from diffstarpop import DEFAULT_DIFFSTARPOP_PARAMS, get_unbounded_diffstarpop_params
from dsps.metallicity import umzr
from jax import jit as jjit
from jax import numpy as jnp

from ..dustpop import tw_dustpop_mono_noise as twdp
from ..ssp_err_model import ssp_err_model
from . import spspop_param_utils as spspu


def get_flat_param_names():
    diffstarpop_pnames_flat = (
        *DEFAULT_DIFFSTARPOP_PARAMS.sfh_pdf_cens_params._fields,
        *DEFAULT_DIFFSTARPOP_PARAMS.satquench_params._fields,
    )

    burstpop_pnames_flat = (
        *spspu.DEFAULT_SPSPOP_PARAMS.burstpop_params.freqburst_params._fields,
        *spspu.DEFAULT_SPSPOP_PARAMS.burstpop_params.fburstpop_params._fields,
        *spspu.DEFAULT_SPSPOP_PARAMS.burstpop_params.tburstpop_params._fields,
    )
    dustpop_pnames_flat = (
        *spspu.DEFAULT_SPSPOP_PARAMS.dustpop_params.avpop_params._fields,
        *spspu.DEFAULT_SPSPOP_PARAMS.dustpop_params.deltapop_params._fields,
        *spspu.DEFAULT_SPSPOP_PARAMS.dustpop_params.funopop_params._fields,
    )
    spspop_pnames_flat = (*burstpop_pnames_flat, *dustpop_pnames_flat)

    all_pnames_flat = (
        *diffstarpop_pnames_flat,
        *umzr.DEFAULT_MZR_PARAMS._fields,
        *spspop_pnames_flat,
        *twdp.DEFAULT_DUSTPOP_SCATTER_PARAMS._fields,
        *ssp_err_model.DEFAULT_SSPERR_PARAMS._fields,
    )
    return all_pnames_flat


@jjit
def unroll_param_collection_into_flat_array(
    diffstarpop_params,
    mzr_params,
    spspop_params,
    dustpop_scatter_params,
    ssp_err_pop_params,
):
    diffstarpop_params_flat = (
        *diffstarpop_params.sfh_pdf_cens_params,
        *diffstarpop_params.satquench_params,
    )

    burstpop_params_flat = (
        *spspop_params.burstpop_params.freqburst_params,
        *spspop_params.burstpop_params.fburstpop_params,
        *spspop_params.burstpop_params.tburstpop_params,
    )
    dustpop_params_flat = (
        *spspop_params.dustpop_params.avpop_params,
        *spspop_params.dustpop_params.deltapop_params,
        *spspop_params.dustpop_params.funopop_params,
    )
    spspop_params_flat = (*burstpop_params_flat, *dustpop_params_flat)

    all_params_flat = (
        *diffstarpop_params_flat,
        *mzr_params,
        *spspop_params_flat,
        *dustpop_scatter_params,
        *ssp_err_pop_params,
    )
    return jnp.array(all_params_flat)


@jjit
def unroll_u_param_collection_into_flat_array(
    diffstarpop_u_params,
    mzr_u_params,
    spspop_u_params,
    dustpop_scatter_u_params,
    ssp_err_pop_u_params,
):
    diffstarpop_u_params_flat = (
        *diffstarpop_u_params.sfh_pdf_cens_params,
        *diffstarpop_u_params.satquench_params,
    )

    burstpop_params_flat = (
        *spspop_u_params.burstpop_params.freqburst_params,
        *spspop_u_params.burstpop_params.fburstpop_params,
        *spspop_u_params.burstpop_params.tburstpop_params,
    )
    dustpop_params_flat = (
        *spspop_u_params.dustpop_params.avpop_params,
        *spspop_u_params.dustpop_params.deltapop_params,
        *spspop_u_params.dustpop_params.funopop_params,
    )
    spspop_u_params_flat = (*burstpop_params_flat, *dustpop_params_flat)

    all_u_params_flat = (
        *diffstarpop_u_params_flat,
        *mzr_u_params,
        *spspop_u_params_flat,
        *dustpop_scatter_u_params,
        *ssp_err_pop_u_params,
    )
    return jnp.array(all_u_params_flat)


@jjit
def get_u_param_collection_from_param_collection(
    diffstarpop_params,
    mzr_params,
    spspop_params,
    dustpop_scatter_params,
    ssp_err_pop_params,
):
    diffstarpop_u_params = get_unbounded_diffstarpop_params(diffstarpop_params)
    mzr_u_params = umzr.get_unbounded_mzr_params(mzr_params)
    spspop_u_params = spspu.get_unbounded_spspop_params_tw_dust(spspop_params)
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
