""""""

from collections import namedtuple

from diffhalos.lightcone import mc_lightcone_halos as mclh
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from jax import numpy as jnp

from ..phot_utils import get_wave_eff_table
from . import precompute_ssp_phot as psspp

N_SFH_TABLE = 100


def mc_weighted_lightcone_data(
    ran_key,
    num_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    ssp_data,
    tcurves,
    z_phot_table,
    logmp_cutoff=11.0,
    cosmo_params=flat_wcdm.PLANCK15,
):
    """
    Generate a mass-function-weighted lightcone of host halos,
    and additional data needed for photometry calculations

    Parameters
    ----------
    ran_key: jran.key
        random key

    num_halos : int
        Number of host halos in the weighted lightcone

    z_min, z_max : float
        min/max redshift

    lgmp_min,lgmp_max : float
        log10 of min/max halo mass in units of Msun

    sky_area_degsq: float
        sky area in deg^2

    ssp_data : namedtuple
        SSP SED templates from DSPS

    tcurves : namedtuple, length (n_bands, )
        each field stores the name of a transmission curve
        each value stores a namedtuple dsps.data_loaders.defaults.TransmissionCurve

    z_phot_table : array, shape (n_z_phot_table, )
        Redshift grid used to tabulate precomputed SSP magnitudes

    hmf_params: namedtuple, optional kwarg
        halo mass function parameters

    logmp_cutoff: float, optional kwarg
        base-10 log of minimum halo mass for which
        DiffmahPop is used to generate MAHs, in Msun;
        for logmp < logmp_cutoff, P(θ_MAH | logmp) = P(θ_MAH | logmp_cutoff)

    cosmo_params: namedtuple, optional kwarg
        cosmological parameters

    Returns
    -------
    lc_data: namedtuple
        Population of num_halos halos along with data needed to compute photometry

            nhalos: ndarray of shape (num_halos, )
                weight of the (sub)halo

            z_obs: ndarray of shape (num_halos, )
                redshift values

            t_obs: ndarray of shape (num_halos, )
                cosmic time at observation, in Gyr

            logmp_obs: ndarray of shape (num_halos, )
                base-10 log of halo mass at observation, in Msun

            mah_params: namedtuple of ndarrays of shape (num_halos, )
                mah parameters

            logmp0: ndarray of shape (num_halos, )
                base-10 log of halo mass at z=0, in Msun

            logt0: float
                Base-10 log of z=0 age of the Universe for the input cosmology

            t_table : array
                Age of the universe in Gyr at which SFH is tabulated

            ssp_data : namedtuple
                same as input

            precomputed_ssp_mag_table : array, shape (n_z_phot_table, n_bands, n_met, n_age)

            z_phot_table : array
                same as input

            wave_eff_table : array, shape (n_z_phot_table, n_bands)
                Effective wavelength of each transmission curve
                evaluated at each redshift in z_phot_table

    """
    args = (ran_key, num_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)
    halopop = mclh.weighted_lc_halos(*args, logmp_cutoff=logmp_cutoff)

    t0 = flat_wcdm.age_at_z0(*cosmo_params)
    t_table = jnp.linspace(T_TABLE_MIN, t0, N_SFH_TABLE)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, cosmo_params
    )
    wave_eff_table = get_wave_eff_table(z_phot_table, tcurves)

    lc_data = LCData(
        halopop.nhalos,
        halopop.z_obs,
        halopop.t_obs,
        halopop.logmp_obs,
        halopop.mah_params,
        halopop.logmp0,
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
    )
    return lc_data


_LCDKEYS = (
    "nhalos",
    "z_obs",
    "t_obs",
    "logmp_obs",
    "mah_params",
    "logmp0",
    "t_table",
    "ssp_data",
    "precomputed_ssp_mag_table",
    "z_phot_table",
    "wave_eff_table",
)
LCData = namedtuple("LCData", _LCDKEYS)
