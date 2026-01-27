from collections import namedtuple
from typing import Callable

import numpy as np
import opencosmo as oc
from diffmah import DiffmahParams
from diffstar import DiffstarParams
from dsps.cosmology import age_at_z

from diffsky import phot_utils
from diffsky.experimental import dbk_phot_from_mock
from diffsky.experimental import precompute_ssp_phot as psspp


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


def __unpack_photometry(data, band_names):
    photometry = data[0].obs_mags.T
    return {name: photometry[i] for i, name in enumerate(band_names)}


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
    insert: bool = True,
):
    print(bands)
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
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    param_collection,
    cosmology,
    Ob0,
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

    result = to_compute(
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
    return unpack_func(result, band_names)
