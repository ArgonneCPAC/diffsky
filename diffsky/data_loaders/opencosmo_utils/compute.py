from collections import namedtuple
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np
import opencosmo as oc
from diffmah import DiffmahParams
from diffstar import DiffstarParams

from ... import phot_utils
from ...experimental import mc_diffstarpop_wrappers as mcdw
from ...experimental import precompute_ssp_phot as psspp
from ...experimental.kernels import (
    dbk_photline_kernels,
    dbk_sed_kernels,
    mc_randoms,
    phot_kernels,
    sed_kernels,
)
from . import utils

DiffstarPopResultsMock = namedtuple(
    "DiffstarPopResultsMock", mcdw.DiffstarPopResults._fields
)


def compute_phot_from_diffsky_mock(
    catalog: oc.Lightcone,
    aux_data: dict,
    bands: list[str],
    insert: bool = True,
    batch_size: int = 50,
):
    """
    Compute photometry for all objects in the catalog for the given bands.

    Parameters:
    -----------

    catalog: opencosmo.Lightcone
        The catalog containing the diffsky data

    aux_data: dict
        The auxilliary data loaded with
        :py:meth:`diffsky.data_loaders.opencosmo_utils.load_diffsky_mock`

    bands: list[str]
        The bands to compute photometry for. You can determine which bands are
        available with :code:`aux_data["tcurves"]._fields`. If you want to compute
        custom bands, first use
        :py:meth:`diffsky.data_loaders.opencosmo_utils.add_transmission_curves`

    insert: bool, default = True
        Whether or not to insert the computed photometry into the dataset. If true,
        returns a dataset with the photometry inserted as new columns. If false, return
        the data directly.

    batch_size: int, default = 50
        The number of systems to compute the photomtery for at once. Note this is NOT
        the number of individual objects, but rather the number of central+satellite
        groups that are computed per pass. If you find yourself running out of memory,
        adjust this number down.

    Returns:
    --------
    data: opencosmo.Lightcone | dict[str, np.ndarray]
        If insert = True, return the new lightcone with the data inserted. If
        insert = False, return the computed photometry directly.



    """
    utils.validate_batch_size(batch_size)
    func = phot_kernels._phot_kern_merging
    z_phot_tables = utils.get_z_phot_tables(catalog)
    suffix = ""
    if set(bands).intersection(catalog.columns):
        suffix = "_new"

    result = __run_photometry(
        func,
        utils.unpack_photometry,
        catalog,
        aux_data,
        z_phot_tables,
        bands,
        None,
        dbk=False,
        insert=False,
        suffix=suffix,
        batch_size=batch_size,
    )
    if insert:
        return catalog.with_new_columns(**result)
    return result


def compute_dbk_phot_from_diffsky_mock(
    catalog: oc.Lightcone,
    aux_data: dict,
    bands: list[str],
    include_extras: Optional[list] = None,
    insert: bool = True,
    batch_size: int = 50,
):
    """
    Compute photometry for all objects in the catalog for the given bands, including
    decomposition into disk/bulge/knot components.

    Parameters:
    -----------

    catalog: opencosmo.Lightcone
        The catalog containing the diffsky data

    aux_data: dict
        The auxilliary data loaded with
        :py:meth:`diffsky.data_loaders.opencosmo_utils.load_diffsky_mock`

    bands: list[str]
        The bands to compute photometry for. You can determine which bands are
        available with :code:`aux_data["tcurves"]._fields`. If you want to compute
        custom bands, first use
        :py:meth:`diffsky.data_loaders.opencosmo_utils.add_transmission_curves`

    insert: bool, default = True
        Whether or not to insert the computed photometry into the dataset. If true,
        returns a dataset with the photometry inserted as new columns. If false, return
        the data directly.
    batch_size: int, default = 50
        The number of systems to compute the photomtery for at once. Note this is NOT
        the number of individual objects, but rather the number of central+satellite
        groups that are computed per pass. If you find yourself running out of memory,
        adjust this number down.

    Returns:
    --------
    data: opencosmo.Lightcone | dict[str, np.ndarray]
        If insert = True, return the new lightcone with the data inserted. If
        insert = False, return the computed photometry directly.

    """
    utils.validate_batch_size(batch_size)

    suffix = ""
    if set(bands).intersection(catalog.columns):
        suffix = "_new"
    z_phot_tables = utils.get_z_phot_tables(catalog)
    func = dbk_photline_kernels._dbk_photline_kern_merging
    result = __run_photometry(
        func,
        utils.unpack_dbk_photometry,
        catalog,
        aux_data,
        z_phot_tables,
        bands,
        include_extras,
        dbk=True,
        insert=False,
        suffix=suffix,
        batch_size=batch_size,
    )

    if insert:
        return catalog.with_new_columns(**result)
    return result


def compute_seds_from_diffsky_mock(
    catalog: oc.Lightcone,
    aux_data: dict,
    insert: bool = True,
    batch_size: int = 25,
):
    """
    Compute SEDs for all objects in the catalog for the given bands.

    Parameters:
    -----------

    catalog: opencosmo.Lightcone
        The catalog containing the diffsky data

    aux_data: dict
        The auxilliary data loaded with
        :py:meth:`diffsky.data_loaders.opencosmo_utils.load_diffsky_mock`

    insert: bool, default = True
        Whether or not to insert the computed photometry into the dataset. If true,
        returns a dataset with the photometry inserted as new columns. If false, return
        the data directly.
    batch_size: int, default = 25
        The number of systems to compute the SED for at once. Note this is NOT
        the number of individual objects, but rather the number of central+satellite
        groups that are computed per pass. If you find yourself running out of memory,
        adjust this number down. Computing SEDs can take significantly more memory
        than photometry.

    Returns:
    --------
    data: opencosmo.Lightcone | dict[str, np.ndarray]
        If insert = True, return the new lightcone with the data inserted. If
        insert = False, return the computed photometry directly.

    """
    utils.validate_batch_size(batch_size)

    result = __run_sed_computation(catalog, aux_data, __compute_sed_managed, batch_size)
    if insert:
        return catalog.with_new_columns(**result)
    return result


def compute_dbk_seds_from_diffsky_mock(
    catalog: oc.Lightcone,
    aux_data: dict,
    insert: bool = True,
    batch_size: int = 25,
):
    """
    Compute SEDs for all objects in the catalog for the given bands, including
    decomposition into disk/bulge/knot components.

    Parameters:
    -----------

    catalog: opencosmo.Lightcone
        The catalog containing the diffsky data

    aux_data: dict
        The auxilliary data loaded with
        :py:meth:`diffsky.data_loaders.opencosmo_utils.load_diffsky_mock`

    insert: bool, default = True
        Whether or not to insert the computed photometry into the dataset. If true,
        returns a dataset with the photometry inserted as new columns. If false, return
        the data directly.
    batch_size: int, default = 25
        The number of systems to compute the SED for at once. Note this is NOT
        the number of individual objects, but rather the number of central+satellite
        groups that are computed per pass. If you find yourself running out of memory,
        adjust this number down. Computing SEDs can take significanty more memory
        than photometry.

    Returns:
    --------
    data: opencosmo.Lightcone | dict[str, np.ndarray]
        If insert = True, return the new lightcone with the data inserted. If
        insert = False, return the computed photometry directly.



    """
    utils.validate_batch_size(batch_size)
    result = __run_sed_computation(
        catalog, aux_data, __compute_dbk_sed_managed, batch_size
    )
    if insert:
        return catalog.with_new_columns(**result)
    return result


def __run_photometry(
    function: Callable,
    unpack_func: Callable,
    catalog: oc.Lightcone,
    aux_data: dict,
    z_phot_tables: dict[str | float, np.ndarray],
    band_names: list[str],
    include_extras: Optional[list],
    dbk: bool,
    insert: bool = True,
    suffix: str = "",
    batch_size: int = 50,
):
    if "tcurves" not in aux_data:
        raise ValueError("Missing transmission curves in auxiliary data!")

    known_tcurves = aux_data["tcurves"]._fields
    missing_tcurves = set(band_names).difference(known_tcurves)
    if missing_tcurves:
        raise ValueError(f"Missing transmission curves for bands {missing_tcurves}")

    Tcurves = namedtuple("Tcurves", tuple(band_names))
    data = {bn: getattr(aux_data["tcurves"], bn) for bn in band_names}
    tcurves = Tcurves(**data)

    wave_eff_tables = {}
    precomputed_ssp_mag_tables = {}
    cosmology_parameters = utils.prep_cosmology_parameters(catalog.cosmology)

    for slice_name, z_phot_table in z_phot_tables.items():
        wave_eff_tables[slice_name] = phot_utils.get_wave_eff_table(
            z_phot_table, tcurves
        )
        precomputed_ssp_mag_tables[slice_name] = (
            psspp.get_precompute_ssp_mag_redshift_table(
                tcurves, aux_data["ssp_data"], z_phot_table, cosmology_parameters
            )
        )
    catalog = catalog.evaluate(
        utils.age_at_z_,
        vectorize=True,
        cosmology=cosmology_parameters,
        format="jax",
    )
    if dbk:
        to_compute = __compute_dbk_photometry_managed
    else:
        to_compute = __compute_photometry_managed

    batches = utils.split_central_indices(catalog, batch_size)

    chunked_output = []
    for row_batch in batches:
        batch_catalog = catalog.take_rows(row_batch)
        batch_ssp_mag_table = {
            k: v
            for k, v in precomputed_ssp_mag_tables.items()
            if k in batch_catalog.keys()
        }
        batch_wave_eff_table = {
            k: v for k, v in wave_eff_tables.items() if k in batch_catalog.keys()
        }
        batch_z_phot_table = {
            k: v for k, v in z_phot_tables.items() if k in batch_catalog.keys()
        }
        batch_output = catalog.take_rows(row_batch).evaluate(
            to_compute,
            to_compute=function,
            unpack_func=unpack_func,
            band_names=band_names,
            cosmology=cosmology_parameters,
            ssp_data=aux_data["ssp_data"],
            precomputed_ssp_mag_table=batch_ssp_mag_table,
            wave_eff_table=batch_wave_eff_table,
            param_collection=aux_data["param_collection"],
            z_phot_table=batch_z_phot_table,
            Ob0=catalog.cosmology.Ob0,
            include_extras=include_extras,
            suffix=suffix,
            insert=insert,
            vectorize=True,
            format="jax",
            # batch_size=batch_size,
        )
        chunked_output.append(batch_output)
    all_bands = chunked_output[0].keys()
    output = {}

    for band in all_bands:
        output[band] = np.concatenate([o[band] for o in chunked_output])

    input_gal_id = catalog.select("gal_id").get_data("numpy")
    output_gal_id = output.pop("gal_id")
    permutation = np.argsort(output_gal_id)[np.argsort(np.argsort(input_gal_id))]

    return {name: band[permutation] for name, band in output.items()}


def __run_sed_computation(
    catalog: oc.Lightcone,
    aux_data: dict,
    to_compute: Callable,
    batch_size: int = 50,
):
    cosmology_parameters = utils.prep_cosmology_parameters(catalog.cosmology)
    catalog = catalog.evaluate(
        utils.age_at_z_,
        vectorize=True,
        cosmology=cosmology_parameters,
        format="jax",
    )

    batches = utils.split_central_indices(catalog, batch_size)

    chunked_output = []
    for row_batch in batches:
        batch_output = catalog.take_rows(row_batch).evaluate(
            to_compute,
            ssp_data=aux_data["ssp_data"],
            param_collection=aux_data["param_collection"],
            cosmology=cosmology_parameters,
            Ob0=catalog.cosmology.Ob0,
            insert=False,
            vectorize=True,
            format="jax",
        )
        chunked_output.append(batch_output)

    all_keys = chunked_output[0].keys()
    output = {}
    for key in all_keys:
        output[key] = np.concatenate([o[key] for o in chunked_output])

    input_gal_id = catalog.select("gal_id").get_data("numpy")
    output_gal_id = output.pop("gal_id")
    permutation = np.argsort(output_gal_id)[np.argsort(np.argsort(input_gal_id))]

    return {name: data[permutation] for name, data in output.items()}


def __compute_sed_managed(
    t_obs,
    mc_sfh_type,
    uran_av,
    uran_delta,
    uran_funo,
    uran_pburst,
    uran_pmerge,
    logm0,
    logtc,
    early_index,
    late_index,
    logmp_infall,
    logmhost_infall,
    central,
    top_host_idx,
    t_peak,  # diffmah params
    lgmcrit,
    lgy_at_mcrit,
    indx_lo,
    indx_hi,  # ms params
    lg_qt,
    qlglgdt,
    lg_drop,
    lg_rejuv,  # Q params
    delta_mag_ssp_scatter,
    redshift_true,
    gal_id,
    ssp_data,
    param_collection,
    cosmology,
    Ob0,
):
    mah_params = DiffmahParams(
        logm0=logm0,
        logtc=logtc,
        early_index=early_index,
        late_index=late_index,
        t_peak=t_peak,
    )
    sfh_params = DiffstarParams(
        lgmcrit=lgmcrit,
        lgy_at_mcrit=lgy_at_mcrit,
        indx_lo=indx_lo,
        indx_hi=indx_hi,
        lg_qt=lg_qt,
        qlglgdt=qlglgdt,
        lg_drop=lg_drop,
        lg_rejuv=lg_rejuv,
    )
    mc_is_q = mc_sfh_type == 0

    phot_randoms = mc_randoms.PhotRandoms(
        mc_is_q, uran_av, uran_delta, uran_funo, uran_pburst, delta_mag_ssp_scatter
    )
    merging_randoms = mc_randoms.DiffMergeRandoms(uran_pmerge)

    sat_weights = jnp.ones(len(redshift_true))
    mc_merge = 1
    args = (
        phot_randoms,
        merging_randoms,
        sfh_params,
        redshift_true,
        t_obs,
        mah_params,
        ssp_data,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
        cosmology,
        Ob0 / cosmology.Om0,
        logmp_infall,
        logmhost_infall,
        t_peak,
        central,
        sat_weights,
        top_host_idx,
        mc_merge,
    )
    sed_info = sed_kernels._sed_kern(*args)
    return {"rest_sed": sed_info.rest_sed, "gal_id": gal_id}


def __compute_dbk_sed_managed(
    t_obs,
    mc_sfh_type,
    fknot,
    uran_fbulge,
    uran_av,
    uran_delta,
    uran_funo,
    uran_pburst,
    uran_pmerge,
    logm0,
    logtc,
    early_index,
    late_index,
    logmp_infall,
    logmhost_infall,
    central,
    top_host_idx,
    t_peak,  # diffmah params
    lgmcrit,
    lgy_at_mcrit,
    indx_lo,
    indx_hi,  # ms params
    lg_qt,
    qlglgdt,
    lg_drop,
    lg_rejuv,  # Q params
    delta_mag_ssp_scatter,
    redshift_true,
    gal_id,
    ssp_data,
    param_collection,
    cosmology,
    Ob0,
):
    mah_params = DiffmahParams(
        logm0=logm0,
        logtc=logtc,
        early_index=early_index,
        late_index=late_index,
        t_peak=t_peak,
    )
    sfh_params = DiffstarParams(
        lgmcrit=lgmcrit,
        lgy_at_mcrit=lgy_at_mcrit,
        indx_lo=indx_lo,
        indx_hi=indx_hi,
        lg_qt=lg_qt,
        qlglgdt=qlglgdt,
        lg_drop=lg_drop,
        lg_rejuv=lg_rejuv,
    )
    mc_is_q = mc_sfh_type == 0

    phot_randoms = mc_randoms.PhotRandoms(
        mc_is_q, uran_av, uran_delta, uran_funo, uran_pburst, delta_mag_ssp_scatter
    )
    dbk_randoms = mc_randoms.DBKRandoms(fknot, uran_fbulge)
    merging_randoms = mc_randoms.DiffMergeRandoms(uran_pmerge)

    sat_weights = jnp.ones(len(redshift_true))
    mc_merge = 1
    args = (
        phot_randoms,
        dbk_randoms,
        merging_randoms,
        sfh_params,
        redshift_true,
        t_obs,
        mah_params,
        ssp_data,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
        cosmology,
        Ob0 / cosmology.Om0,
        logmp_infall,
        logmhost_infall,
        t_peak,
        central,
        sat_weights,
        top_host_idx,
        mc_merge,
    )
    sed_info = dbk_sed_kernels._dbk_sed_kern(*args)
    return {
        "rest_sed_bulge": sed_info.rest_sed_bulge,
        "rest_sed_disk": sed_info.rest_sed_disk,
        "rest_sed_knots": sed_info.rest_sed_knots,
        "gal_id": gal_id,
    }


def __compute_photometry_managed(
    to_compute,
    unpack_func,
    band_names,
    logm0,
    logtc,
    early_index,
    late_index,
    logmp_infall,
    logmhost_infall,
    central,
    t_peak,  # diffmah params
    t_obs,
    lgmcrit,
    lgy_at_mcrit,
    indx_lo,
    indx_hi,  # ms params
    lg_qt,
    uran_pmerge,
    qlglgdt,
    lg_drop,
    lg_rejuv,  # Q params
    uran_av,
    uran_delta,
    uran_funo,
    uran_pburst,
    delta_mag_ssp_scatter,  # randoms
    redshift_true,
    top_host_idx,
    mc_sfh_type,
    ssp_data,
    gal_id,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    param_collection,
    cosmology,
    Ob0,
    include_extras,
    suffix="",
):
    mc_is_q = mc_sfh_type == 0
    mah_params = DiffmahParams(
        logm0=logm0,
        logtc=logtc,
        early_index=early_index,
        late_index=late_index,
        t_peak=t_peak,
    )
    sfh_params = DiffstarParams(
        lgmcrit=lgmcrit,
        lgy_at_mcrit=lgy_at_mcrit,
        indx_lo=indx_lo,
        indx_hi=indx_hi,
        lg_qt=lg_qt,
        qlglgdt=qlglgdt,
        lg_drop=lg_drop,
        lg_rejuv=lg_rejuv,
    )
    dummy_frac_q = jnp.ones(mc_is_q.size)
    diffstarpop_results_mock = DiffstarPopResultsMock(
        sfh_params, sfh_params, sfh_params, mc_is_q, dummy_frac_q
    )

    phot_randoms = mc_randoms.PhotRandoms(
        mc_is_q, uran_av, uran_delta, uran_funo, uran_pburst, delta_mag_ssp_scatter
    )
    merging_randoms = mc_randoms.DiffMergeRandoms(uran_pmerge)

    sat_weights = jnp.ones(len(redshift_true))
    mc_merge = 1
    args = (
        phot_randoms,
        merging_randoms,
        diffstarpop_results_mock,
        redshift_true,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
        cosmology,
        Ob0 / cosmology.Om0,
        logmp_infall,
        logmhost_infall,
        t_peak,
        central,
        sat_weights,
        top_host_idx,
        mc_merge,
    )
    result = to_compute(*args)
    return unpack_func(result, band_names, suffix, gal_id, include_extras)


def __compute_dbk_photometry_managed(
    to_compute,
    unpack_func,
    band_names,
    logm0,
    logtc,
    early_index,
    late_index,
    logmp_infall,
    logmhost_infall,
    central,
    top_host_idx,
    t_peak,  # diffmah params
    lgmcrit,
    lgy_at_mcrit,
    indx_lo,
    indx_hi,  # ms params
    lg_qt,
    qlglgdt,
    lg_drop,
    lg_rejuv,  # Q params
    uran_av,
    uran_delta,
    uran_funo,
    uran_pburst,
    uran_fbulge,
    uran_pmerge,
    delta_mag_ssp_scatter,  # randoms
    redshift_true,
    t_obs,
    mc_sfh_type,
    fknot,
    ssp_data,
    gal_id,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    param_collection,
    cosmology,
    Ob0,
    include_extras,
    suffix="",
):
    mc_is_q = mc_sfh_type == 0
    mah_params = DiffmahParams(
        logm0=logm0,
        logtc=logtc,
        early_index=early_index,
        late_index=late_index,
        t_peak=t_peak,
    )
    sfh_params = DiffstarParams(
        lgmcrit=lgmcrit,
        lgy_at_mcrit=lgy_at_mcrit,
        indx_lo=indx_lo,
        indx_hi=indx_hi,
        lg_qt=lg_qt,
        qlglgdt=qlglgdt,
        lg_drop=lg_drop,
        lg_rejuv=lg_rejuv,
    )
    dummy_frac_q = jnp.ones(mc_is_q.size)
    diffstarpop_results_mock = DiffstarPopResultsMock(
        sfh_params, sfh_params, sfh_params, mc_is_q, dummy_frac_q
    )

    phot_randoms = mc_randoms.PhotRandoms(
        mc_is_q, uran_av, uran_delta, uran_funo, uran_pburst, delta_mag_ssp_scatter
    )
    dbk_randoms = mc_randoms.DBKRandoms(fknot, uran_fbulge)
    merging_randoms = mc_randoms.DiffMergeRandoms(uran_pmerge)
    line_wave_table = jnp.array(ssp_data.ssp_emline_wave)
    sat_weights = jnp.ones(len(t_obs))
    mc_merge = 1

    args = (
        phot_randoms,
        diffstarpop_results_mock,
        dbk_randoms,
        merging_randoms,
        redshift_true,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        line_wave_table,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
        cosmology,
        Ob0 / cosmology.Om0,
        logmp_infall,
        logmhost_infall,
        t_peak,
        central,
        sat_weights,
        top_host_idx,
        mc_merge,
    )
    result = to_compute(*args)
    return unpack_func(result, band_names, suffix, gal_id, include_extras)
