""""""

from diffmah import DEFAULT_MAH_PARAMS
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology.flat_wcdm import age_at_z
from jax import jit as jjit
from jax import numpy as jnp

from diffsky import phot_utils
from diffsky.experimental import precompute_ssp_phot as psspp
from diffsky.experimental.dbk_from_mock import _disk_bulge_knot_phot_from_mock


def get_ssp_phot_tables(tcurves, z_phot_table, ssp_data, sim_info):

    wave_eff_table = phot_utils.get_wave_eff_table(z_phot_table, tcurves)
    ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, sim_info.cosmo_params
    )
    return ssp_mag_table, wave_eff_table


def compute_mock_photometry(mock_data, mock_info, ssp_mag_table, wave_eff_table):
    mah_params = DEFAULT_MAH_PARAMS._make(
        [mock_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [mock_data[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    t_obs = age_at_z(mock_data["redshift_true"], *mock_info["sim_info"].cosmo_params)

    delta_scatter = jnp.where(
        mock_data["mc_sfh_type"].reshape((-1, 1)) == 0,
        mock_data["delta_scatter_q"],
        mock_data["delta_scatter_ms"],
    )

    args = (
        mock_data["redshift_true"],
        t_obs,
        mah_params,
        mock_data["logmp0"],
        mock_info["t_table"],
        mock_info["ssp_data"],
        ssp_mag_table,
        mock_info["z_phot_table"],
        wave_eff_table,
        sfh_params,
        mock_info["param_collection"].mzr_params,
        mock_info["param_collection"].spspop_params,
        mock_info["param_collection"].scatter_params,
        mock_info["param_collection"].ssperr_params,
        mock_info["sim_info"].cosmo_params,
        mock_info["sim_info"].fb,
        mock_data["uran_av"],
        mock_data["uran_delta"],
        mock_data["uran_funo"],
        delta_scatter,
        mock_data["mc_sfh_type"],
        mock_data["fknot"],
    )
    phot_data = _disk_bulge_knot_phot_from_mock(*args)

    return phot_data
