""""""

from jax import jit as jjit


@jjit
def _reproduce_mock_dbk_merging_kern(
    mc_is_q,
    uran_av,
    uran_delta,
    uran_funo,
    uran_pburst,
    delta_mag_ssp_scatter,
    sfh_params,
    z_obs,
    t_obs,
    mah_params,
    fknot,
    uran_fbulge,
    ssp_data,
    precomputed_ssp_mag_table,
    z_phot_table,
    wave_eff_table,
    mzr_params,
    spspop_params,
    scatter_params,
    ssp_err_pop_params,
    cosmo_params,
    fb,
):
    pass
