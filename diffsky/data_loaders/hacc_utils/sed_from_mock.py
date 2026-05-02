""""""

from jax import numpy as jnp
from diffmah import DEFAULT_MAH_PARAMS
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology.flat_wcdm import age_at_z
from dsps.sfh.diffburst import DEFAULT_BURST_PARAMS
from collections import namedtuple

from ... import phot_utils
from ...experimental import dbk_phot_from_mock, dbk_phot_from_mock_merging
from ...experimental import precompute_ssp_phot as psspp
from ...experimental.kernels import (
    dbk_kernels,
    dbk_specphot_kernels,
    mc_randoms,
    sed_kernels_merging,
)


def compute_phot_from_diffsky_mock_merging(lc_mock_chunk, metadata, tcurves=None):
    if tcurves is None:
        tcurves = metadata["tcurves"]

    param_collection = metadata["param_collection"]
    mah_params = DEFAULT_MAH_PARAMS._make(
        [lc_mock_chunk[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [lc_mock_chunk[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    t_obs = age_at_z(lc_mock_chunk["redshift_true"], *metadata["sim_info"].cosmo_params)

    wave_eff_table = phot_utils.get_wave_eff_table(metadata["z_phot_table"], tcurves)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves,
        metadata["ssp_data"],
        metadata["z_phot_table"],
        metadata["sim_info"].cosmo_params,
    )
    n_chunk = len(t_obs)
    sat_weights = jnp.ones(n_chunk)
    halo_indx = lc_mock_chunk["top_host_idx_chunk"]
    t_infall = lc_mock_chunk["t_peak"]

    args = (
        lc_mock_chunk["mc_sfh_type"],
        lc_mock_chunk["uran_av"],
        lc_mock_chunk["uran_delta"],
        lc_mock_chunk["uran_funo"],
        lc_mock_chunk["uran_pburst"],
        lc_mock_chunk["delta_mag_ssp_scatter"],
        lc_mock_chunk["uran_pmerge"],
        sfh_params,
        lc_mock_chunk["redshift_true"],
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
        lc_mock_chunk["logmp_infall"],
        lc_mock_chunk["logmhost_infall"],
        t_infall,
        lc_mock_chunk["central"],
        sat_weights,
        halo_indx,
    )
    phot_kern_results = dbk_phot_from_mock_merging._reproduce_mock_phot_kern(*args)
    return phot_kern_results


def compute_phot_from_diffsky_mock(
    *, diffsky_data, ssp_data, param_collection, sim_info, z_phot_table, tcurves
):
    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    t_obs = age_at_z(diffsky_data["redshift_true"], *sim_info.cosmo_params)

    wave_eff_table = phot_utils.get_wave_eff_table(z_phot_table, tcurves)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, sim_info.cosmo_params
    )

    mc_is_q = jnp.where(diffsky_data["mc_sfh_type"] == 0, True, False)
    args = (
        mc_is_q,
        diffsky_data["uran_av"],
        diffsky_data["uran_delta"],
        diffsky_data["uran_funo"],
        diffsky_data["uran_pburst"],
        diffsky_data["delta_mag_ssp_scatter"],
        sfh_params,
        diffsky_data["redshift_true"],
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
        sim_info.cosmo_params,
        sim_info.fb,
    )

    phot_info = dbk_phot_from_mock._reproduce_mock_phot_kern(*args)[0]
    phot_info = phot_info._asdict()
    return phot_info


def compute_dbk_phot_from_diffsky_mock(
    *, diffsky_data, ssp_data, param_collection, sim_info, z_phot_table, tcurves
):
    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    t_obs = age_at_z(diffsky_data["redshift_true"], *sim_info.cosmo_params)

    wave_eff_table = phot_utils.get_wave_eff_table(z_phot_table, tcurves)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, sim_info.cosmo_params
    )

    mc_is_q = jnp.where(diffsky_data["mc_sfh_type"] == 0, True, False)
    args = (
        mc_is_q,
        diffsky_data["uran_av"],
        diffsky_data["uran_delta"],
        diffsky_data["uran_funo"],
        diffsky_data["uran_pburst"],
        diffsky_data["delta_mag_ssp_scatter"],
        sfh_params,
        diffsky_data["redshift_true"],
        t_obs,
        mah_params,
        diffsky_data["fknot"],
        diffsky_data["uran_fbulge"],
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        sim_info.cosmo_params,
        sim_info.fb,
    )
    _res = dbk_phot_from_mock._reproduce_mock_dbk_kern(*args)
    (
        phot_info,
        phot_randoms,
        disk_bulge_history,
        obs_mags_bulge,
        obs_mags_disk,
        obs_mags_knots,
    ) = _res
    phot_info = phot_info._asdict()
    phot_info["obs_mags_bulge"] = obs_mags_bulge
    phot_info["obs_mags_disk"] = obs_mags_disk
    phot_info["obs_mags_knots"] = obs_mags_knots
    return phot_info


def compute_dbk_phot_from_diffsky_mock_merging(lc_mock_chunk, metadata, tcurves=None):
    if tcurves is None:
        tcurves = metadata["tcurves"]

    param_collection = metadata["param_collection"]
    mah_params = DEFAULT_MAH_PARAMS._make(
        [lc_mock_chunk[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [lc_mock_chunk[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    t_obs = age_at_z(lc_mock_chunk["redshift_true"], *metadata["sim_info"].cosmo_params)

    wave_eff_table = phot_utils.get_wave_eff_table(metadata["z_phot_table"], tcurves)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves,
        metadata["ssp_data"],
        metadata["z_phot_table"],
        metadata["sim_info"].cosmo_params,
    )
    n_chunk = len(t_obs)
    sat_weights = jnp.ones(n_chunk)
    halo_indx = lc_mock_chunk["top_host_idx_chunk"]
    t_infall = lc_mock_chunk["t_peak"]

    args = (
        lc_mock_chunk["mc_sfh_type"],
        lc_mock_chunk["uran_av"],
        lc_mock_chunk["uran_delta"],
        lc_mock_chunk["uran_funo"],
        lc_mock_chunk["uran_pburst"],
        lc_mock_chunk["delta_mag_ssp_scatter"],
        lc_mock_chunk["uran_fbulge"],
        lc_mock_chunk["fknot"],
        lc_mock_chunk["uran_pmerge"],
        sfh_params,
        lc_mock_chunk["redshift_true"],
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
        lc_mock_chunk["logmp_infall"],
        lc_mock_chunk["logmhost_infall"],
        t_infall,
        lc_mock_chunk["central"],
        sat_weights,
        halo_indx,
    )
    phot_kern_results = dbk_phot_from_mock_merging._reproduce_dbk_mock_phot_kern(*args)
    return phot_kern_results


def compute_sed_from_diffsky_mock(
    *, diffsky_data, ssp_data, param_collection, sim_info, z_phot_table, tcurves
):
    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    t_obs = age_at_z(diffsky_data["redshift_true"], *sim_info.cosmo_params)

    wave_eff_table = phot_utils.get_wave_eff_table(z_phot_table, tcurves)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, sim_info.cosmo_params
    )

    mc_is_q = jnp.where(diffsky_data["mc_sfh_type"] == 0, True, False)
    args = (
        mc_is_q,
        diffsky_data["uran_av"],
        diffsky_data["uran_delta"],
        diffsky_data["uran_funo"],
        diffsky_data["uran_pburst"],
        diffsky_data["delta_mag_ssp_scatter"],
        sfh_params,
        diffsky_data["redshift_true"],
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
        sim_info.cosmo_params,
        sim_info.fb,
    )

    phot_info, __, sed_kern_results = dbk_phot_from_mock._reproduce_mock_sed_kern(*args)
    sed_info = phot_info._asdict()
    rest_sed = sed_kern_results[0]
    sed_info["rest_sed"] = rest_sed
    return sed_info


def compute_sed_from_diffsky_mock_merging(
    *, diffsky_data, ssp_data, param_collection, sim_info
):
    # wave_eff_table = phot_utils.get_wave_eff_table(metadata["z_phot_table"], tcurves)

    # precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
    #     tcurves,
    #     metadata["ssp_data"],
    #     metadata["z_phot_table"],
    #     metadata["sim_info"].cosmo_params,
    # )

    t_obs = age_at_z(diffsky_data["redshift_true"], *sim_info.cosmo_params)

    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )

    # precomputed_ssp_mag_table.shape = (n_gals, n_bands, n_met, n_age)
    n_bands = 1
    n_gals = diffsky_data["redshift_true"].size
    n_met = ssp_data.ssp_lgmet.size
    n_age = ssp_data.ssp_lg_age_gyr.size
    precomputed_ssp_mag_table = jnp.ones((n_gals, n_bands, n_met, n_age))

    # wave_eff_table.shape = (n_z_phot_table, n_bands)
    z_min = diffsky_data["redshift_true"].min()
    z_max = diffsky_data["redshift_true"].max()
    dz = z_max - z_min
    eps = dz / 100.0
    z_phot_table = jnp.array((z_min - eps, z_max + eps))
    wave_eff_table = jnp.ones((z_phot_table.size, n_bands))

    mc_is_q = jnp.where(diffsky_data["mc_sfh_type"] == 0, True, False)
    args = (
        mc_is_q,
        diffsky_data["uran_av"],
        diffsky_data["uran_delta"],
        diffsky_data["uran_funo"],
        diffsky_data["uran_pburst"],
        diffsky_data["delta_mag_ssp_scatter"],
        sfh_params,
        diffsky_data["redshift_true"],
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
        sim_info.cosmo_params,
        sim_info.fb,
    )

    phot_info = dbk_phot_from_mock_merging._reproduce_mock_phot_kern(*args)[0]

    args = (
        phot_randoms,
        merging_randoms,
        sfh_params,
        z_obs,
        t_obs,
        mah_params,
        ssp_data,
        mzr_params,
        spspop_params,
        scatter_params,
        ssp_err_pop_params,
        merging_params,
        cosmo_params,
        fb,
        logmp_infall,
        logmhost_infall,
        t_infall,
        is_central,
        sat_weights,
        halo_indx,
        mc_merge,
    )
    sed_results = sed_kernels_merging._sed_kern(*args)


def compute_dbk_sed_from_diffsky_mock(
    *, diffsky_data, ssp_data, param_collection, sim_info, z_phot_table, tcurves
):
    dbk_phot_info = compute_dbk_phot_from_diffsky_mock(
        diffsky_data=diffsky_data,
        ssp_data=ssp_data,
        param_collection=param_collection,
        sim_info=sim_info,
        z_phot_table=z_phot_table,
        tcurves=tcurves,
    )
    t_obs = age_at_z(diffsky_data["redshift_true"], *sim_info.cosmo_params)

    dbk_randoms = mc_randoms.DBKRandoms(
        diffsky_data["fknot"], diffsky_data["uran_fbulge"]
    )

    dbk_phot_info["uran_av"] = diffsky_data["uran_av"]
    dbk_phot_info["uran_delta"] = diffsky_data["uran_delta"]
    dbk_phot_info["uran_funo"] = diffsky_data["uran_funo"]
    dbk_phot_info["delta_mag_ssp_scatter"] = diffsky_data["delta_mag_ssp_scatter"]

    burst_params = DEFAULT_BURST_PARAMS._make(
        [dbk_phot_info[pname] for pname in DEFAULT_BURST_PARAMS._fields]
    )

    args = (
        t_obs,
        ssp_data,
        dbk_phot_info["t_table"],
        dbk_phot_info["sfh_table"],
        burst_params,
        dbk_phot_info["lgmet_weights"],
        dbk_randoms,
    )
    dbk_weights, disk_bulge_history = dbk_kernels._dbk_kern(*args)

    dbk_phot_info["mstar_bulge"] = dbk_weights.mstar_bulge
    dbk_phot_info["mstar_disk"] = dbk_weights.mstar_disk
    dbk_phot_info["mstar_knots"] = dbk_weights.mstar_knots

    DBKPhotInfo = namedtuple("DBKPhotInfo", list(dbk_phot_info.keys()))
    dbk_phot_info = DBKPhotInfo(**dbk_phot_info)

    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffsky_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )

    mc_is_q = jnp.where(diffsky_data["mc_sfh_type"] == 0, True, False)
    sed_info, __ = dbk_specphot_kernels._dbk_sed_kern(
        mc_is_q,
        diffsky_data["uran_av"],
        diffsky_data["uran_delta"],
        diffsky_data["uran_funo"],
        diffsky_data["uran_pburst"],
        diffsky_data["delta_mag_ssp_scatter"],
        diffsky_data["uran_fbulge"],
        diffsky_data["fknot"],
        sfh_params,
        diffsky_data["redshift_true"],
        t_obs,
        mah_params,
        ssp_data,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        sim_info.cosmo_params,
        sim_info.fb,
    )
    dbk_sed_info = dbk_phot_info._asdict()
    dbk_sed_info["rest_sed_bulge"] = sed_info.rest_sed_bulge
    dbk_sed_info["rest_sed_disk"] = sed_info.rest_sed_disk
    dbk_sed_info["rest_sed_knots"] = sed_info.rest_sed_knots
    return dbk_sed_info
