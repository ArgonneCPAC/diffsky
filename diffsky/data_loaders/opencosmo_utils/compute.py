from collections import namedtuple
from typing import Callable, Optional

import numpy as np
import opencosmo as oc
from diffmah import DiffmahParams
from diffstar import DiffstarParams
from dsps.cosmology import age_at_z
from dsps.sfh.diffburst import BurstParams

from ... import phot_utils
from ...experimental import dbk_phot_from_mock
from ...experimental import precompute_ssp_phot as psspp
from ...experimental.kernels import mc_phot_kernels as mcpk


def __get_z_phot_tables(catalog: oc.Lightcone):
    """
    In the future, this method will iterate through the catalogs, retrieve the min+max
    z_phot, and construct a redshift-slice-specific z_phot table. For the moment,
    it just looks at the min and max redshift of the slice, widens it a bit,
    and constructs a table. Beta software stuff.
    """
    z_phot_tables = {}
    for slice_name, dataset in catalog.items():
        if isinstance(dataset, oc.Dataset):
            min_z, max_z = dataset.header.lightcone["z_range"]
        elif isinstance(dataset, oc.Lightcone):
            min_z, max_z = dataset.z_range

        if min_z != 0.0:
            min_z = 0.95 * min_z
        max_z = 1.05 * max_z
        z_phot_tables[slice_name] = np.linspace(min_z, max_z, 15)
    return z_phot_tables


def compute_phot_from_diffsky_mock(
    catalog: oc.Lightcone,
    aux_data: dict,
    bands: list[str],
    insert: bool = True,
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

    z_phot_table:
        Not sure how we want to handle this one (yet)

    bands: list[str]
        The bands to compute photometry for. You can determine which bands are
        available with :code:`aux_data["tcurves"]._fields`. If you want to compute
        custom bands, first use
        :py:meth:`diffsky.data_loaders.opencosmo_utils.add_transmission_curves`

    insert: bool, default = True
        Whether or not to insert the computed photometry into the dataset. If true,
        returns a dataset with the photometry inserted as new columns. If false, return
        the data directly.

    Returns:
    --------
    data: opencosmo.Lightcone | dict[str, np.ndarray]
        If insert = True, return the new lightcone with the data inserted. If
        insert = False, return the computed photometry directly.



    """
    func = dbk_phot_from_mock._reproduce_mock_phot_kern
    z_phot_tables = __get_z_phot_tables(catalog)
    suffix = ""
    if set(bands).intersection(catalog.columns):
        suffix = "_new"

    result = __run_photometry(
        func,
        __unpack_photometry,
        catalog,
        aux_data,
        z_phot_tables,
        bands,
        None,
        False,
        insert=False,
        suffix=suffix,
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

    z_phot_table:
        Not sure how we want to handle this one (yet)

    bands: list[str]
        The bands to compute photometry for. You can determine which bands are
        available with :code:`aux_data["tcurves"]._fields`. If you want to compute
        custom bands, first use
        :py:meth:`diffsky.data_loaders.opencosmo_utils.add_transmission_curves`

    insert: bool, default = True
        Whether or not to insert the computed photometry into the dataset. If true,
        returns a dataset with the photometry inserted as new columns. If false, return
        the data directly.

    Returns:
    --------
    data: opencosmo.Lightcone | dict[str, np.ndarray]
        If insert = True, return the new lightcone with the data inserted. If
        insert = False, return the computed photometry directly.



    """

    suffix = ""
    if set(bands).intersection(catalog.columns):
        suffix = "_new"
    z_phot_tables = __get_z_phot_tables(catalog)
    func = dbk_phot_from_mock._reproduce_mock_dbk_kern
    result = __run_photometry(
        func,
        __unpack_dbk_photometry,
        catalog,
        aux_data,
        z_phot_tables,
        bands,
        include_extras,
        True,
        insert=False,
        suffix=suffix,
    )
    if insert:
        return catalog.with_new_columns(**result)
    return result


def compute_seds_from_diffsky_mock(
    catalog: oc.Lightcone,
    aux_data: dict,
    bands: list[str],
    insert: bool = True,
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

    z_phot_table:
        Not sure how we want to handle this one (yet)

    bands: list[str]
        The bands to compute photometry for. You can determine which bands are
        available with :code:`aux_data["tcurves"]._fields`. If you want to compute
        custom bands, first use
        :py:meth:`diffsky.data_loaders.opencosmo_utils.add_transmission_curves`

    insert: bool, default = True
        Whether or not to insert the computed photometry into the dataset. If true,
        returns a dataset with the photometry inserted as new columns. If false, return
        the data directly.

    Returns:
    --------
    data: opencosmo.Lightcone | dict[str, np.ndarray]
        If insert = True, return the new lightcone with the data inserted. If
        insert = False, return the computed photometry directly.



    """

    z_phot_tables = __get_z_phot_tables(catalog)
    func = dbk_phot_from_mock._reproduce_mock_sed_kern
    result = __run_photometry(
        func,
        __unpack_seds,
        catalog,
        aux_data,
        z_phot_tables,
        bands,
        None,
        False,
        insert=False,
    )
    if insert:
        return catalog.with_new_columns(result)
    return result


def compute_dbk_seds_from_diffsky_mock(
    catalog: oc.Lightcone,
    aux_data: dict,
    bands: list[str],
    insert: bool = True,
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

    z_phot_table:
        Not sure how we want to handle this one (yet)

    bands: list[str]
        The bands to compute photometry for. You can determine which bands are
        available with :code:`aux_data["tcurves"]._fields`. If you want to compute
        custom bands, first use
        :py:meth:`diffsky.data_loaders.opencosmo_utils.add_transmission_curves`

    insert: bool, default = True
        Whether or not to insert the computed photometry into the dataset. If true,
        returns a dataset with the photometry inserted as new columns. If false, return
        the data directly.

    Returns:
    --------
    data: opencosmo.Lightcone | dict[str, np.ndarray]
        If insert = True, return the new lightcone with the data inserted. If
        insert = False, return the computed photometry directly.



    """
    z_phot_tables = __get_z_phot_tables(catalog)
    cosmology_parameters = __prep_cosmology_parameters(catalog.cosmology)
    dbk_phot_info = compute_dbk_phot_from_diffsky_mock(
        catalog,
        aux_data,
        bands,
        ["t_table", "sfh_table", "lgmet_weights"],
        False,
    )
    catalog = catalog.evaluate(
        age_at_z_, vectorize=True, cosmology=cosmology_parameters
    )
    result = catalog.evaluate(
        __compute_dbk_sed_managed,
        dbk_phot_info=dbk_phot_info,
        ssp_data=aux_data["ssp_data"],
        param_collection=aux_data["param_collection"],
        cosmology=cosmology_parameters,
        insert=False,
        vectorize=True,
    )
    if insert is True:
        return catalog.with_new_columns(**result)
    return result


def __unpack_photometry(data, band_names, suffix, *args):
    return __unpack_photometry_array(data[0].obs_mags, band_names, suffix)


def __unpack_dbk_photometry(data, band_names, suffix, include_extras):
    (phot_info, _, _, obs_mag_bulge, obs_mag_disk, obs_mag_knots) = data
    bulge_bands = [f"{bn}_bulge" for bn in band_names]
    disk_bands = [f"{bn}_disk" for bn in band_names]
    knot_bands = [f"{bn}_knots" for bn in band_names]

    output = __unpack_photometry_array(obs_mag_bulge, bulge_bands, suffix)
    output |= __unpack_photometry_array(obs_mag_disk, disk_bands, suffix)
    output |= __unpack_photometry_array(obs_mag_knots, knot_bands, suffix)
    if include_extras is not None:
        phot_info = phot_info._asdict()
        output |= {name: phot_info[name] for name in include_extras}

    return output


def __unpack_photometry_array(data, band_names, suffix):
    to_unpack = np.array(data).T
    return {f"{name}{suffix}": to_unpack[i] for i, name in enumerate(band_names)}


def __unpack_seds(data, band_names, *args):
    phot_info, _, sed_kern_results = data
    rest_sed = sed_kern_results[0]
    return {"rest_sed": rest_sed}


def __run_photometry(
    function: Callable,
    unpack_func: Callable,
    catalog: oc.Lightcone,
    aux_data: dict,
    z_phot_tables: dict[str | float, np.ndarray],
    band_names: list[str],
    include_extras: Optional[list],
    do_decomp: bool = False,
    insert: bool = True,
    suffix: str = "",
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

    cosmology_parameters = __prep_cosmology_parameters(catalog.cosmology)
    wave_eff_tables = {}
    precomputed_ssp_mag_tables = {}
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
        age_at_z_, vectorize=True, cosmology=cosmology_parameters
    )
    return catalog.evaluate(
        __compute_photometry_managed,
        to_compute=function,
        unpack_func=unpack_func,
        band_names=band_names,
        cosmology=cosmology_parameters,
        ssp_data=aux_data["ssp_data"],
        precomputed_ssp_mag_table=precomputed_ssp_mag_tables,
        wave_eff_table=wave_eff_tables,
        param_collection=aux_data["param_collection"],
        z_phot_table=z_phot_tables,
        Ob0=catalog.cosmology.Ob0,
        include_extras=include_extras,
        do_decomp=do_decomp,
        suffix=suffix,
        insert=insert,
        vectorize=True,
        format="numpy",
    )


def __prep_cosmology_parameters(cosmology):
    try:
        w0 = cosmology.wo
        wa = cosmology.wa
    except AttributeError:  # Why astropy... Why...
        w0 = -1
        wa = 0
    Cosmology_t = namedtuple("Cosmology_t", ("Om0", "w0", "wa", "h"))
    return Cosmology_t(cosmology.Om0, w0, wa, cosmology.h)


def age_at_z_(redshift_true, cosmology):
    result = {
        "t_obs": np.array(
            age_at_z(
                redshift_true, cosmology.Om0, cosmology.w0, cosmology.wa, cosmology.h
            )
        )
    }
    return result


def __compute_dbk_sed_managed(
    t_obs,
    fknot,
    uran_av,
    uran_delta,
    uran_funo,
    lgfburst,
    lgyr_peak,
    lgyr_max,
    logsm_obs,
    logssfr_obs,
    delta_mag_ssp_scatter,
    redshift_true,
    dbk_phot_info,
    ssp_data,
    param_collection,
    cosmology,
    suffix="",
):
    burst_params = BurstParams(
        lgfburst=lgfburst, lgyr_peak=lgyr_peak, lgyr_max=lgyr_max
    )

    DBKRandoms = namedtuple("DBKRandoms", ("fknot",))
    dbk_randoms = DBKRandoms(fknot)

    dbk_phot_info["uran_av"] = uran_av
    dbk_phot_info["uran_delta"] = uran_delta
    dbk_phot_info["uran_funo"] = uran_funo
    dbk_phot_info["delta_mag_ssp_scatter"] = delta_mag_ssp_scatter
    dbk_phot_info["logsm_obs"] = logsm_obs
    dbk_phot_info["logssfr_obs"] = logssfr_obs

    args = (
        t_obs,
        ssp_data,
        dbk_phot_info["t_table"],
        dbk_phot_info["sfh_table"],
        burst_params,
        dbk_phot_info["lgmet_weights"],
        dbk_randoms,
    )

    dbk_weights, disk_bulge_history = mcpk._dbk_kern(*args)
    dbk_phot_info["mstar_bulge"] = dbk_weights.mstar_bulge
    dbk_phot_info["mstar_disk"] = dbk_weights.mstar_disk
    dbk_phot_info["mstar_knots"] = dbk_weights.mstar_knots
    DBKPhotInfo = namedtuple("DBKPhotInfo", list(dbk_phot_info.keys()))
    dbk_phot_info = DBKPhotInfo(**dbk_phot_info)

    sed_bulge, sed_disk, sed_knots = mcpk._mc_lc_dbk_sed_kern(
        dbk_phot_info,
        dbk_weights,
        redshift_true,
        ssp_data,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
    )

    dbk_sed_info = dbk_phot_info._asdict()
    dbk_sed_info["rest_sed_bulge"] = sed_bulge
    dbk_sed_info["rest_sed_disk"] = sed_disk
    dbk_sed_info["rest_sed_knots"] = sed_knots
    return dbk_sed_info


def __compute_photometry_managed(
    to_compute,
    unpack_func,
    band_names,
    logm0,
    logtc,
    early_index,
    late_index,
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
    uran_pburst,  # ¯\_(ツ)_/¯ (I am not a real astrophysicist)
    delta_mag_ssp_scatter,
    redshift_true,
    t_obs,
    mc_sfh_type,
    fknot,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    param_collection,
    cosmology,
    Ob0,
    include_extras,
    do_decomp=False,
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

    if do_decomp:
        fknot_arg = (fknot,)
    else:
        fknot_arg = ()

    args = (
        mc_is_q,
        uran_av,
        uran_delta,
        uran_funo,
        uran_pburst,
        delta_mag_ssp_scatter,
        sfh_params,
        redshift_true,
        t_obs,
        mah_params,
        *fknot_arg,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        cosmology,
        Ob0 / cosmology.Om0,
    )
    result = to_compute(*args)
    return unpack_func(result, band_names, suffix, include_extras)
