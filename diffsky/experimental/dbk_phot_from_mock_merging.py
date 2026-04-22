""""""

from jax import jit as jjit
from jax import numpy as jnp

from .kernels import dbk_specphot_kernels_merging as dbkspkm
from .kernels import mc_randoms, phot_kernels_merging


@jjit
def _reproduce_mock_phot_kern(
    mc_sfh_type,
    uran_av,
    uran_delta,
    uran_funo,
    uran_pburst,
    delta_mag_ssp_scatter,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    merge_params,
    cosmo_params,
    fb,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    nhalos_weights,
    halo_indx,
):
    mc_is_q = mc_sfh_type == 0
    phot_randoms = mc_randoms.PhotRandoms(
        mc_is_q,
        uran_av,
        uran_delta,
        uran_funo,
        uran_pburst,
        delta_mag_ssp_scatter,
    )

    phot_kern_results, phot_randoms = phot_kernels_merging._phot_kern_merging(
        phot_randoms,
        sfh_params,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        merge_params,
        cosmo_params,
        fb,
        logmp_infall,
        logmhost_infall,
        t_infall,
        is_central,
        nhalos_weights,
        halo_indx,
    )
    return phot_kern_results


@jjit
def _reproduce_dbk_mock_phot_kern(
    mc_sfh_type,
    uran_av,
    uran_delta,
    uran_funo,
    uran_pburst,
    delta_mag_ssp_scatter,
    uran_fbulge,
    fknot,
    uran_pmerge,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    merge_params,
    cosmo_params,
    fb,
    logmp_infall,
    logmhost_infall,
    t_infall,
    is_central,
    nhalos_weights,
    halo_indx,
):
    mc_is_q = mc_sfh_type == 0
    phot_randoms = mc_randoms.PhotRandoms(
        mc_is_q, uran_av, uran_delta, uran_funo, uran_pburst, delta_mag_ssp_scatter
    )

    dbk_randoms = mc_randoms.DBKRandoms(fknot=fknot, uran_fbulge=uran_fbulge)
    merging_randoms = mc_randoms.DiffMergeRandoms(uran_pmerge)
    line_wave_table = jnp.array(ssp_data.ssp_emline_wave)

    mc_merge = 1
    dbk_specphot_info, dbk_weights = dbkspkm._dbk_specphot_kern_merging(
        phot_randoms,
        sfh_params,
        dbk_randoms,
        merging_randoms,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        line_wave_table,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        merge_params,
        cosmo_params,
        fb,
        logmp_infall,
        logmhost_infall,
        t_infall,
        is_central,
        nhalos_weights,
        halo_indx,
        mc_merge,
    )
    return dbk_specphot_info, dbk_weights
