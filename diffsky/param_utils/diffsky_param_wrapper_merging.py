""" """

from collections import namedtuple

import numpy as np
from diffstar.diffstarpop import (
    DEFAULT_DIFFSTARPOP_PARAMS,
    DEFAULT_DIFFSTARPOP_U_PARAMS,
    get_bounded_diffstarpop_params,
    get_unbounded_diffstarpop_params,
)
from dsps.metallicity import umzr
from jax import jit as jjit

from ..experimental.scatter import (
    DEFAULT_SCATTER_PARAMS,
    DEFAULT_SCATTER_U_PARAMS,
    get_bounded_scatter_params,
    get_unbounded_scatter_params,
)
from ..merging import merging_model
from ..ssp_err_model import ssp_err_model
from . import spspop_param_utils as spspu

ParamCollection = namedtuple(
    "ParamCollection",
    (
        "diffstarpop_params",
        "mzr_params",
        "spspop_params",
        "scatter_params",
        "ssperr_params",
        "merging_params",
    ),
)
DEFAULT_PARAM_COLLECTION = ParamCollection(
    DEFAULT_DIFFSTARPOP_PARAMS,
    umzr.DEFAULT_MZR_PARAMS,
    spspu.DEFAULT_SPSPOP_PARAMS,
    DEFAULT_SCATTER_PARAMS,
    ssp_err_model.DEFAULT_SSPERR_PARAMS,
    merging_model.DEFAULT_MERGE_PARAMS,
)
UParamCollection = namedtuple(
    "UParamCollection",
    (
        "diffstarpop_u_params",
        "mzr_u_params",
        "spspop_u_params",
        "scatter_u_params",
        "ssperr_u_params",
        "merging_u_params",
    ),
)


def get_flat_param_names():
    diffstarpop_pnames_flat = (*DEFAULT_DIFFSTARPOP_PARAMS._fields,)

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
        *DEFAULT_SCATTER_PARAMS._fields,
        *ssp_err_model.DEFAULT_SSPERR_PARAMS._fields,
        *merging_model.DEFAULT_MERGE_PARAMS._fields,
    )
    return all_pnames_flat


@jjit
def unroll_param_collection_into_flat_array(
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    merging_params,
):
    diffstarpop_params_flat = diffstarpop_params

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

    diffsky_params_flat = DiffskyParamsFlat(
        *diffstarpop_params_flat,
        *mzr_params,
        *spspop_params_flat,
        *scatter_params,
        *ssp_err_pop_params,
        *merging_params,
    )
    return diffsky_params_flat


@jjit
def get_param_collection_from_flat_array(all_params_flat):
    all_pnames_flat = get_flat_param_names()
    DiffskyParams = namedtuple("DiffskyParams", all_pnames_flat)
    named_params = DiffskyParams(*all_params_flat)

    diffstarpop_params = DEFAULT_DIFFSTARPOP_PARAMS._make(
        [getattr(named_params, pname) for pname in DEFAULT_DIFFSTARPOP_PARAMS._fields]
    )
    mzr_params = [
        getattr(named_params, pname) for pname in umzr.DEFAULT_MZR_PARAMS._fields
    ]
    mzr_params = umzr.DEFAULT_MZR_PARAMS._make(mzr_params)

    freqburst_params = spspu.DEFAULT_SPSPOP_PARAMS.burstpop_params.freqburst_params._make(
        [
            getattr(named_params, pname)
            for pname in spspu.DEFAULT_SPSPOP_PARAMS.burstpop_params.freqburst_params._fields
        ]
    )
    fburstpop_params = spspu.DEFAULT_SPSPOP_PARAMS.burstpop_params.fburstpop_params._make(
        [
            getattr(named_params, pname)
            for pname in spspu.DEFAULT_SPSPOP_PARAMS.burstpop_params.fburstpop_params._fields
        ]
    )
    tburstpop_params = spspu.DEFAULT_SPSPOP_PARAMS.burstpop_params.tburstpop_params._make(
        [
            getattr(named_params, pname)
            for pname in spspu.DEFAULT_SPSPOP_PARAMS.burstpop_params.tburstpop_params._fields
        ]
    )
    burstpop_params = spspu.DEFAULT_SPSPOP_PARAMS.burstpop_params._make(
        (freqburst_params, fburstpop_params, tburstpop_params)
    )

    avpop_params = spspu.DEFAULT_SPSPOP_PARAMS.dustpop_params.avpop_params._make(
        [
            getattr(named_params, pname)
            for pname in spspu.DEFAULT_SPSPOP_PARAMS.dustpop_params.avpop_params._fields
        ]
    )

    deltapop_params = spspu.DEFAULT_SPSPOP_PARAMS.dustpop_params.deltapop_params._make(
        [
            getattr(named_params, pname)
            for pname in spspu.DEFAULT_SPSPOP_PARAMS.dustpop_params.deltapop_params._fields
        ]
    )

    funopop_params = spspu.DEFAULT_SPSPOP_PARAMS.dustpop_params.funopop_params._make(
        [
            getattr(named_params, pname)
            for pname in spspu.DEFAULT_SPSPOP_PARAMS.dustpop_params.funopop_params._fields
        ]
    )
    dustpop_params = spspu.DEFAULT_SPSPOP_PARAMS.dustpop_params._make(
        (avpop_params, deltapop_params, funopop_params)
    )
    spspop_params = spspu.DEFAULT_SPSPOP_PARAMS._make((burstpop_params, dustpop_params))

    scatter_params = [
        getattr(named_params, pname) for pname in DEFAULT_SCATTER_PARAMS._fields
    ]
    scatter_params = DEFAULT_SCATTER_PARAMS._make(scatter_params)

    ssp_err_params = [
        getattr(named_params, pname)
        for pname in ssp_err_model.DEFAULT_SSPERR_PARAMS._fields
    ]
    ssp_err_params = ssp_err_model.DEFAULT_SSPERR_PARAMS._make(ssp_err_params)

    merging_params = [
        getattr(named_params, pname)
        for pname in merging_model.DEFAULT_MERGE_PARAMS._fields
    ]
    merging_params = merging_model.DEFAULT_MERGE_PARAMS._make(merging_params)

    param_collection = ParamCollection(
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_params,
        merging_params,
    )

    return param_collection


@jjit
def unroll_u_param_collection_into_flat_array(
    diffstarpop_u_params,
    mzr_u_params,
    spspop_u_params,
    scatter_u_params,
    ssp_err_pop_u_params,
    merging_u_params,
):
    diffstarpop_u_params_flat = diffstarpop_u_params

    burstpop_params_flat = (
        *spspop_u_params.u_burstpop_params.freqburst_u_params,
        *spspop_u_params.u_burstpop_params.fburstpop_u_params,
        *spspop_u_params.u_burstpop_params.tburstpop_u_params,
    )
    dustpop_params_flat = (
        *spspop_u_params.u_dustpop_params.avpop_u_params,
        *spspop_u_params.u_dustpop_params.deltapop_u_params,
        *spspop_u_params.u_dustpop_params.funopop_u_params,
    )
    spspop_u_params_flat = (*burstpop_params_flat, *dustpop_params_flat)

    diffsky_u_params_flat = DiffskyUParamsFlat(
        *diffstarpop_u_params_flat,
        *mzr_u_params,
        *spspop_u_params_flat,
        *scatter_u_params,
        *ssp_err_pop_u_params,
        *merging_u_params,
    )
    return diffsky_u_params_flat


@jjit
def get_u_param_collection_from_param_collection(
    diffstarpop_params,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    merging_params,
):
    diffstarpop_u_params = get_unbounded_diffstarpop_params(diffstarpop_params)
    mzr_u_params = umzr.get_unbounded_mzr_params(mzr_params)
    spspop_u_params = spspu.get_unbounded_spspop_params_tw_dust(spspop_params)
    scatter_u_params = get_unbounded_scatter_params(scatter_params)
    ssp_err_pop_u_params = ssp_err_model.get_unbounded_ssperr_params(ssp_err_pop_params)
    merging_u_params = merging_model.get_unbounded_merge_params(merging_params)

    u_param_collection = UParamCollection(
        diffstarpop_u_params,
        mzr_u_params,
        spspop_u_params,
        scatter_u_params,
        ssp_err_pop_u_params,
        merging_u_params,
    )
    return u_param_collection


@jjit
def get_param_collection_from_u_param_collection(
    diffstarpop_u_params,
    mzr_u_params,
    spspop_u_params,
    scatter_u_params,
    ssp_err_pop_u_params,
    merging_u_params,
):
    diffstarpop_params = get_bounded_diffstarpop_params(diffstarpop_u_params)
    mzr_params = umzr.get_bounded_mzr_params(mzr_u_params)
    spspop_params = spspu.get_bounded_spspop_params_tw_dust(spspop_u_params)
    scatter_params = get_bounded_scatter_params(scatter_u_params)
    ssp_err_pop_params = ssp_err_model.get_bounded_ssperr_params(ssp_err_pop_u_params)
    merging_params = merging_model.get_bounded_merge_params(merging_u_params)

    param_collection = ParamCollection(
        diffstarpop_params,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        merging_params,
    )
    return param_collection


@jjit
def get_u_param_collection_from_u_param_array(u_param_arr):
    u_params = DiffskyUParamsFlat(*u_param_arr)

    diffstarpop_u_params = DEFAULT_DIFFSTARPOP_U_PARAMS._make(
        [getattr(u_params, name) for name in DEFAULT_DIFFSTARPOP_U_PARAMS._fields]
    )

    u_mzr_params = [
        getattr(u_params, name) for name in umzr.DEFAULT_MZR_U_PARAMS._fields
    ]
    u_mzr_params = umzr.DEFAULT_MZR_U_PARAMS._make(u_mzr_params)

    freqburst_u_params = [
        getattr(u_params, name)
        for name in spspu.DEFAULT_SPSPOP_U_PARAMS.u_burstpop_params.freqburst_u_params._fields
    ]
    freqburst_u_params = (
        spspu.DEFAULT_SPSPOP_U_PARAMS.u_burstpop_params.freqburst_u_params._make(
            freqburst_u_params
        )
    )
    fburstpop_u_params = [
        getattr(u_params, name)
        for name in spspu.DEFAULT_SPSPOP_U_PARAMS.u_burstpop_params.fburstpop_u_params._fields
    ]
    fburstpop_u_params = (
        spspu.DEFAULT_SPSPOP_U_PARAMS.u_burstpop_params.fburstpop_u_params._make(
            fburstpop_u_params
        )
    )
    tburstpop_u_params = [
        getattr(u_params, name)
        for name in spspu.DEFAULT_SPSPOP_U_PARAMS.u_burstpop_params.tburstpop_u_params._fields
    ]
    tburstpop_u_params = (
        spspu.DEFAULT_SPSPOP_U_PARAMS.u_burstpop_params.tburstpop_u_params._make(
            tburstpop_u_params
        )
    )
    u_burstpop_params = (freqburst_u_params, fburstpop_u_params, tburstpop_u_params)
    u_burstpop_params = spspu.DEFAULT_SPSPOP_U_PARAMS.u_burstpop_params._make(
        u_burstpop_params
    )
    avpop_u_params = [
        getattr(u_params, name)
        for name in spspu.DEFAULT_SPSPOP_U_PARAMS.u_dustpop_params.avpop_u_params._fields
    ]
    avpop_u_params = (
        spspu.DEFAULT_SPSPOP_U_PARAMS.u_dustpop_params.avpop_u_params._make(
            avpop_u_params
        )
    )

    deltapop_u_params = [
        getattr(u_params, name)
        for name in spspu.DEFAULT_SPSPOP_U_PARAMS.u_dustpop_params.deltapop_u_params._fields
    ]
    deltapop_u_params = (
        spspu.DEFAULT_SPSPOP_U_PARAMS.u_dustpop_params.deltapop_u_params._make(
            deltapop_u_params
        )
    )

    funopop_u_params = [
        getattr(u_params, name)
        for name in spspu.DEFAULT_SPSPOP_U_PARAMS.u_dustpop_params.funopop_u_params._fields
    ]
    funopop_u_params = (
        spspu.DEFAULT_SPSPOP_U_PARAMS.u_dustpop_params.funopop_u_params._make(
            funopop_u_params
        )
    )

    u_dustpop_params = (avpop_u_params, deltapop_u_params, funopop_u_params)
    u_dustpop_params = spspu.DEFAULT_SPSPOP_U_PARAMS.u_dustpop_params._make(
        u_dustpop_params
    )
    spspop_u_params = spspu.DEFAULT_SPSPOP_U_PARAMS._make(
        (u_burstpop_params, u_dustpop_params)
    )

    scatter_u_params = [
        getattr(u_params, name) for name in DEFAULT_SCATTER_U_PARAMS._fields
    ]
    scatter_u_params = DEFAULT_SCATTER_U_PARAMS._make(scatter_u_params)

    ssp_err_pop_u_params = [
        getattr(u_params, name)
        for name in ssp_err_model.DEFAULT_SSPERR_U_PARAMS._fields
    ]
    ssp_err_pop_u_params = ssp_err_model.DEFAULT_SSPERR_U_PARAMS._make(
        ssp_err_pop_u_params
    )

    merging_u_params = [
        getattr(u_params, name) for name in merging_model.DEFAULT_MERGE_U_PARAMS._fields
    ]
    merging_u_params = merging_model.DEFAULT_MERGE_U_PARAMS._make(merging_u_params)

    u_param_collection = UParamCollection(
        diffstarpop_u_params,
        u_mzr_params,
        spspop_u_params,
        scatter_u_params,
        ssp_err_pop_u_params,
        merging_u_params,
    )
    return u_param_collection


PNAMES_FLAT = get_flat_param_names()
DiffskyParamsFlat = namedtuple("DiffskyParamsFlat", PNAMES_FLAT)
U_PNAMES_FLAT = ["u_" + name for name in PNAMES_FLAT]
DiffskyUParamsFlat = namedtuple("DiffskyUParamsFlat", U_PNAMES_FLAT)


def check_param_collection_is_ok(param_collection):
    """Check the input param_collection can be unbounded and rebounded without issues"""
    param_collection_is_ok = True
    diffsky_params_flat = unroll_param_collection_into_flat_array(*param_collection)

    if not np.all(np.isfinite(diffsky_params_flat)):
        param_collection_is_ok = False
        print("Some non-finite values in param_collection")

    u_param_collection = get_u_param_collection_from_param_collection(*param_collection)
    diffsky_u_params_flat = unroll_u_param_collection_into_flat_array(
        *u_param_collection
    )
    if not np.all(np.isfinite(diffsky_u_params_flat)):
        param_collection_is_ok = False
        print("Some non-finite values in unbounded param_collection")

    param_collection2 = get_param_collection_from_u_param_collection(
        *u_param_collection
    )
    diffsky_params2_flat = unroll_param_collection_into_flat_array(*param_collection2)
    if not np.allclose(diffsky_params_flat, diffsky_params2_flat, rtol=1e-3):
        param_collection_is_ok = False
        print("param_collection is not the same after unbounding and rebounding")

    return param_collection_is_ok
