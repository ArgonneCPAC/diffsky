from collections import namedtuple
from typing import Callable, Optional

import numpy as np
import opencosmo as oc
from diffmah import DiffmahParams
from diffstar import DiffstarParams
from dsps.cosmology import age_at_z
from dsps.sfh.diffburst import BurstParams

from diffsky import phot_utils
from diffsky.experimental import dbk_phot_from_mock
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.experimental.kernels import mc_phot_kernels as mcpk


def compute_phot_from_diffsky_mocks(
    catalog: oc.Lightcone,
    aux_data: dict,
    z_phot_table: np.ndarray,
    survey_name: str = "lsst",
    bands: list[str] = ["u", "g", "r", "i", "z", "y"],
    insert: bool = True,
):
    func = dbk_phot_from_mock._reproduce_mock_phot_kern
    return __run_photometry(
        func,
        __unpack_photometry,
        catalog,
        aux_data,
        z_phot_table,
        survey_name,
        bands,
        None,
        insert,
    )


def compute_dbk_phot_from_diffsky_mocks(
    catalog: oc.Lightcone,
    aux_data: dict,
    z_phot_table: np.ndarray,
    survey_name: str = "lsst",
    bands: list[str] = ["u", "g", "r", "i", "z", "y"],
    include_extras: Optional[list] = None,
    insert: bool = True,
):
    func = dbk_phot_from_mock._reproduce_mock_dbk_kern
    return __run_photometry(
        func,
        __unpack_dbk_photometry,
        catalog,
        aux_data,
        z_phot_table,
        survey_name,
        bands,
        include_extras,
        True,
        insert,
    )


def compute_seds_from_diffsky_mocks(
    catalog: oc.Lightcone,
    aux_data: dict,
    z_phot_table: np.ndarray,
    survey_name: str = "lsst",
    bands: list[str] = ["u", "g", "r", "i", "z", "y"],
    insert: bool = True,
):
    func = dbk_phot_from_mock._reproduce_mock_sed_kern
    return __run_photometry(
        func, __unpack_seds, catalog, aux_data, z_phot_table, survey_name, bands, insert
    )


def compute_dbk_seds_from_diffsky_mocks(
    catalog: oc.Lightcone,
    aux_data: dict,
    z_phot_table: np.ndarray,
    survey_name: str = "lsst",
    bands: list[str] = ["u", "g", "r", "i", "z", "y"],
    insert: bool = True,
):
    cosmology_parameters = prep_cosmology_parameters(catalog.cosmology)
    dbk_phot_info = compute_dbk_phot_from_diffsky_mocks(
        catalog,
        aux_data,
        z_phot_table,
        survey_name,
        bands,
        ["t_table", "sfh_table", "lgmet_weights"],
        False,
    )
    catalog = catalog.evaluate(
        age_at_z_, vectorize=True, cosmology=cosmology_parameters
    )
    return catalog.evaluate(
        compute_dbk_sed_managed,
        dbk_phot_info=dbk_phot_info,
        ssp_data=aux_data["ssp_data"],
        param_collection=aux_data["param_collection"],
        cosmology=cosmology_parameters,
        insert=insert,
        vectorize=True,
    )


def __unpack_photometry(data, band_names):
    return __unpack_photometry_array(data[0].obs_mags)


def __unpack_dbk_photometry(data, band_names, include_extras):
    (phot_info, _, _, obs_mag_bulge, obs_mag_disk, obs_mag_knots) = data
    bulge_bands = [f"{bn}_bulge" for bn in band_names]
    disk_bands = [f"{bn}_disk" for bn in band_names]
    knot_bands = [f"{bn}_knots" for bn in band_names]

    output = __unpack_photometry_array(obs_mag_bulge, bulge_bands)
    output |= __unpack_photometry_array(obs_mag_disk, disk_bands)
    output |= __unpack_photometry_array(obs_mag_knots, knot_bands)
    if include_extras is not None:
        phot_info = phot_info._asdict()
        output |= {name: phot_info[name] for name in include_extras}

    return output


def __unpack_photometry_array(data, band_names):
    to_unpack = np.array(data).T
    return {name: to_unpack[i] for i, name in enumerate(band_names)}


def __unpack_seds(data, band_names):
    phot_info, _, sed_kern_results = data
    sed_info = phot_info._asdict()
    rest_sed = sed_kern_results[0]
    sed_info["rest_sed"] = rest_sed
    return sed_info


def __run_photometry(
    function: Callable,
    unpack_func: Callable,
    catalog: oc.Lightcone,
    aux_data: dict,
    z_phot_table: np.ndarray,
    survey_name: str,
    bands: list[str],
    include_extras: Optional[list],
    do_decomp: bool = False,
    insert: bool = True,
):
    band_names = set(map(lambda band: f"{survey_name}_{band}", bands))
    if "tcurves" not in aux_data:
        raise ValueError("Missing transmission curves in auxiliary data!")

    known_tcurves = aux_data["tcurves"]._fields
    missing_tcurves = band_names.difference(known_tcurves)
    if missing_tcurves:
        raise ValueError(f"Missing transmission curves for bands {missing_tcurves}")

    Tcurves = namedtuple("Tcurves", tuple(band_names))
    data = {bn: getattr(aux_data["tcurves"], bn) for bn in band_names}
    tcurves = Tcurves(**data)

    wave_eff_table = phot_utils.get_wave_eff_table(z_phot_table, tcurves)
    cosmology_parameters = prep_cosmology_parameters(catalog.cosmology)
    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, aux_data["ssp_data"], z_phot_table, cosmology_parameters
    )
    catalog = catalog.evaluate(
        age_at_z_, vectorize=True, cosmology=cosmology_parameters
    )
    return catalog.evaluate(
        compute_photometry_managed,
        to_compute=function,
        unpack_func=unpack_func,
        band_names=band_names,
        cosmology=cosmology_parameters,
        ssp_data=aux_data["ssp_data"],
        precomputed_ssp_mag_table=precomputed_ssp_mag_table,
        wave_eff_table=wave_eff_table,
        param_collection=aux_data["param_collection"],
        z_phot_table=z_phot_table,
        Ob0=catalog.cosmology.Ob0,
        include_extras=include_extras,
        do_decomp=do_decomp,
        insert=insert,
        vectorize=True,
        format="numpy",
    )


def prep_cosmology_parameters(cosmology):
    try:
        w0 = cosmology.wo
        wa = cosmology.wa
    except AttributeError:  # Why astropy... Why...
        w0 = -1
        wa = 0
    Cosmology_t = namedtuple("Cosmology_t", ("Om0", "w0", "wa", "h"))
    return Cosmology_t(cosmology.Om0, w0, wa, cosmology.h)


def age_at_z_(redshift, cosmology):
    return {
        "t_obs": np.array(
            age_at_z(redshift, cosmology.Om0, cosmology.w0, cosmology.wa, cosmology.h)
        )
    }


def compute_dbk_sed_managed(
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
    redshift,
    dbk_phot_info,
    ssp_data,
    param_collection,
    cosmology,
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
        redshift,
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


def compute_photometry_managed(
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
    redshift,
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
        redshift,
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
    return unpack_func(result, band_names, include_extras)
