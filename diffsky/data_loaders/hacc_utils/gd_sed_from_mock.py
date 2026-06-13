""""""

from collections import namedtuple

import numpy as np
from diffmah import DEFAULT_MAH_PARAMS
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology.flat_wcdm import age_at_z
from jax import numpy as jnp

from ... import phot_utils
from ...experimental import mc_diffstarpop_wrappers as mcdw
from ...experimental import precompute_ssp_phot as psspp
from ...experimental.kernels import (
    dbk_sed_kernels,
)
from ...experimental.kernels import dbk_specphot_kernels_merging as gd_dbkspkm
from ...experimental.kernels import (
    phot_kernels_merging,
    sed_kernels_merging,
    mc_randoms,
)

DiffstarPopResultsMock = namedtuple(
    "DiffstarPopResultsMock", mcdw.DiffstarPopResults._fields
)


def compute_phot_from_mock(
    mock_chunk, metadata, tcurves=None, precomputed_ssp_mag_table=None
):
    if tcurves is None:
        tcurves = metadata["tcurves"]

    param_collection = metadata["param_collection"]
    mah_params = DEFAULT_MAH_PARAMS._make(
        [mock_chunk[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [mock_chunk[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    mc_is_q = mock_chunk["mc_sfh_type"] == 0
    dummy_frac_q = np.zeros(mc_is_q.size) + 0.5
    diffstarpop_results_mock = DiffstarPopResultsMock(
        sfh_params, sfh_params, sfh_params, mc_is_q, dummy_frac_q
    )
    t_obs = age_at_z(mock_chunk["redshift_true"], *metadata["sim_info"].cosmo_params)

    wave_eff_table = phot_utils.get_wave_eff_table(metadata["z_phot_table"], tcurves)

    n_chunk = len(t_obs)
    sat_weights = jnp.ones(n_chunk)
    halo_indx = mock_chunk["top_host_idx_chunk"]
    t_infall = mock_chunk["t_peak"]

    if precomputed_ssp_mag_table is None:
        precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
            tcurves,
            metadata["ssp_data"],
            metadata["z_phot_table"],
            metadata["sim_info"].cosmo_params,
        )

    mc_is_q = mock_chunk["mc_sfh_type"] == 0
    phot_randoms = mc_randoms.PhotRandoms(
        mc_is_q,
        mock_chunk["uran_av"],
        mock_chunk["uran_delta"],
        mock_chunk["uran_funo"],
        mock_chunk["uran_pburst"],
        mock_chunk["delta_mag_ssp_scatter"],
    )
    merging_randoms = mc_randoms.DiffMergeRandoms(mock_chunk["uran_pmerge"])
    mc_merge = 1
    args = (
        phot_randoms,
        merging_randoms,
        diffstarpop_results_mock,
        mock_chunk["redshift_true"],
        t_obs,
        mah_params,
        metadata["ssp_data"],
        precomputed_ssp_mag_table,
        metadata["z_phot_table"],
        wave_eff_table,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
        metadata["sim_info"].cosmo_params,
        metadata["sim_info"].fb,
        mock_chunk["logmp_infall"],
        mock_chunk["logmhost_infall"],
        t_infall,
        mock_chunk["central"],
        sat_weights,
        halo_indx,
        mc_merge,
    )
    phot_info = phot_kernels_merging._phot_kern_merging(*args)
    phot_info = phot_info._asdict()
    phot_info.update(phot_randoms._asdict())
    phot_info.update(merging_randoms._asdict())
    return phot_info


def compute_dbk_phot_from_mock(mock_chunk, metadata, tcurves=None):
    if tcurves is None:
        tcurves = metadata["tcurves"]

    param_collection = metadata["param_collection"]
    mah_params = DEFAULT_MAH_PARAMS._make(
        [mock_chunk[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [mock_chunk[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    mc_is_q = mock_chunk["mc_sfh_type"] == 0
    dummy_frac_q = np.zeros(mc_is_q.size) + 0.5
    diffstarpop_results_mock = DiffstarPopResultsMock(
        sfh_params, sfh_params, sfh_params, mc_is_q, dummy_frac_q
    )

    t_obs = age_at_z(mock_chunk["redshift_true"], *metadata["sim_info"].cosmo_params)

    wave_eff_table = phot_utils.get_wave_eff_table(metadata["z_phot_table"], tcurves)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves,
        metadata["ssp_data"],
        metadata["z_phot_table"],
        metadata["sim_info"].cosmo_params,
    )
    n_chunk = len(t_obs)
    sat_weights = jnp.ones(n_chunk)
    halo_indx = mock_chunk["top_host_idx_chunk"]
    t_infall = mock_chunk["t_peak"]

    mc_is_q = mock_chunk["mc_sfh_type"] == 0
    phot_randoms = mc_randoms.PhotRandoms(
        mc_is_q,
        mock_chunk["uran_av"],
        mock_chunk["uran_delta"],
        mock_chunk["uran_funo"],
        mock_chunk["uran_pburst"],
        mock_chunk["delta_mag_ssp_scatter"],
    )

    dbk_randoms = mc_randoms.DBKRandoms(
        fknot=mock_chunk["fknot"], uran_fbulge=mock_chunk["uran_fbulge"]
    )
    merging_randoms = mc_randoms.DiffMergeRandoms(mock_chunk["uran_pmerge"])
    line_wave_table = jnp.array(metadata["ssp_data"].ssp_emline_wave)

    mc_merge = 1

    args = (
        phot_randoms,
        diffstarpop_results_mock,
        dbk_randoms,
        merging_randoms,
        mock_chunk["redshift_true"],
        t_obs,
        mah_params,
        metadata["ssp_data"],
        precomputed_ssp_mag_table,
        metadata["z_phot_table"],
        wave_eff_table,
        line_wave_table,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        param_collection.merging_params,
        metadata["sim_info"].cosmo_params,
        metadata["sim_info"].fb,
        mock_chunk["logmp_infall"],
        mock_chunk["logmhost_infall"],
        t_infall,
        mock_chunk["central"],
        sat_weights,
        halo_indx,
        mc_merge,
    )

    dbk_phot_info, dbk_weights = gd_dbkspkm._dbk_specphot_kern_merging(*args)
    dbk_phot_info = dbk_phot_info._asdict()
    dbk_weights = dbk_weights._asdict()
    return dbk_phot_info, dbk_weights


def compute_sed_from_mock(mock_chunk, metadata):

    t_obs = age_at_z(mock_chunk["redshift_true"], *metadata["sim_info"].cosmo_params)

    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [mock_chunk[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    mah_params = DEFAULT_MAH_PARAMS._make(
        [mock_chunk[key] for key in DEFAULT_MAH_PARAMS._fields]
    )

    t_infall = mock_chunk["t_peak"]
    n_chunk = len(t_obs)
    sat_weights = jnp.ones(n_chunk)
    halo_indx = mock_chunk["top_host_idx_chunk"]

    mc_is_q = jnp.where(mock_chunk["mc_sfh_type"] == 0, True, False)
    phot_randoms = mc_randoms.PhotRandoms(
        mc_is_q,
        mock_chunk["uran_av"],
        mock_chunk["uran_delta"],
        mock_chunk["uran_funo"],
        mock_chunk["uran_pburst"],
        mock_chunk["delta_mag_ssp_scatter"],
    )
    merging_randoms = mc_randoms.DiffMergeRandoms(mock_chunk["uran_pmerge"])

    mc_merge = 1
    args = (
        phot_randoms,
        merging_randoms,
        sfh_params,
        mock_chunk["redshift_true"],
        t_obs,
        mah_params,
        metadata["ssp_data"],
        metadata["param_collection"].mzr_params,
        metadata["param_collection"].spspop_params,
        metadata["param_collection"].scatter_params,
        metadata["param_collection"].ssperr_params,
        metadata["param_collection"].merging_params,
        metadata["sim_info"].cosmo_params,
        metadata["sim_info"].fb,
        mock_chunk["logmp_infall"],
        mock_chunk["logmhost_infall"],
        t_infall,
        mock_chunk["central"],
        sat_weights,
        halo_indx,
        mc_merge,
    )
    sed_info = sed_kernels_merging._sed_kern(*args)
    sed_info = sed_info._asdict()
    return sed_info


def compute_dbk_sed_from_mock(mock_chunk, metadata):
    t_obs = age_at_z(mock_chunk["redshift_true"], *metadata["sim_info"].cosmo_params)

    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [mock_chunk[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    mah_params = DEFAULT_MAH_PARAMS._make(
        [mock_chunk[key] for key in DEFAULT_MAH_PARAMS._fields]
    )

    t_infall = mock_chunk["t_peak"]
    n_chunk = len(t_obs)
    sat_weights = jnp.ones(n_chunk)
    halo_indx = mock_chunk["top_host_idx_chunk"]

    mc_is_q = jnp.where(mock_chunk["mc_sfh_type"] == 0, True, False)
    phot_randoms = mc_randoms.PhotRandoms(
        mc_is_q,
        mock_chunk["uran_av"],
        mock_chunk["uran_delta"],
        mock_chunk["uran_funo"],
        mock_chunk["uran_pburst"],
        mock_chunk["delta_mag_ssp_scatter"],
    )
    merging_randoms = mc_randoms.DiffMergeRandoms(mock_chunk["uran_pmerge"])
    dbk_randoms = mc_randoms.DBKRandoms(
        fknot=mock_chunk["fknot"], uran_fbulge=mock_chunk["uran_fbulge"]
    )

    mc_merge = 1

    args = (
        phot_randoms,
        dbk_randoms,
        merging_randoms,
        sfh_params,
        mock_chunk["redshift_true"],
        t_obs,
        mah_params,
        metadata["ssp_data"],
        metadata["param_collection"].mzr_params,
        metadata["param_collection"].spspop_params,
        metadata["param_collection"].scatter_params,
        metadata["param_collection"].ssperr_params,
        metadata["param_collection"].merging_params,
        metadata["sim_info"].cosmo_params,
        metadata["sim_info"].fb,
        mock_chunk["logmp_infall"],
        mock_chunk["logmhost_infall"],
        t_infall,
        mock_chunk["central"],
        sat_weights,
        halo_indx,
        mc_merge,
    )
    dbk_sed_info = dbk_sed_kernels._dbk_sed_kern(
        *args, n_t_table=mcdw.N_T_TABLE
    )
    dbk_sed_info = dbk_sed_info._asdict()
    return dbk_sed_info
