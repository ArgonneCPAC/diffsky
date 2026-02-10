"""Wrappers around diffhalos.lightcone_generators that computes additional data
need to calculate photometry and run gradient descents
"""

from collections import namedtuple

from diffhalos.lightcone_generators import mc_lightcone as mcl
from diffhalos.lightcone_generators import mc_lightcone_halos as mclh
from dsps.constants import T_TABLE_MIN
from dsps.cosmology import flat_wcdm
from jax import numpy as jnp

from ..phot_utils import get_wave_eff_table
from . import precompute_ssp_phot as psspp

N_SFH_TABLE = 100


def weighted_lc_halos_photdata(
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
    Generate a weighted lightcone of host halos,
    and additional data needed for photometry calculations.

    This function is a wrapper around
    diffhalos.lightcone_generators.mc_lightcone_halos.weighted_lc_halos

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

    lc_data = passively_add_emlines_to_lc_data(ssp_data, lc_data)

    return lc_data


def weighted_lc_photdata(
    ran_key,
    n_host_halos,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    ssp_data,
    tcurves,
    z_phot_table,
    *,
    cosmo_params=flat_wcdm.PLANCK15,
    logmp_cutoff=11.0,
):
    """
    Generate a weighted lightcone of host halos,
    and additional data needed for photometry calculations.

    This function is a wrapper around
    diffhalos.lightcone_generators.mc_lightcone.weighted_lc

    Parameters
    ----------
    ran_key: jran.key
        random key

    n_host_halos : int
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
        Population of n_halos_tot halos along with data needed to compute photometry

    halopop: namedtuple
        Population of n_halos_tot halos and subhalos
            n_halos_tot = n_sub + n_host_halos
            n_sub = nsub_per_host * n_host_halos

        halopop fields:
            z_obs: ndarray of shape (n_halos_tot, )
                redshift values

            t_obs: ndarray of shape (n_halos_tot, )
                cosmic time at observation, in Gyr

            logmp_obs: ndarray of shape (n_halos_tot, )
                base-10 log of halo mass at observation, in Msun

            mah_params: namedtuple of ndarrays of shape (n_halos_tot, )
                mah parameters

            logmp0: ndarray of shape (n_halos_tot, )
                base-10 log of halo mass at z=0, in Msun

            logt0: float
                Base-10 log of z=0 age of the Universe for the input cosmology

            nhalos: ndarray of shape (n_halos_tot, )
                weight of the (sub)halo

            nhalos_host: ndarray of shape (n_halos_tot, )
                weight of the host halo
                Equal to nhalos for central halos

            nsub_per_host: int
                number of subhalos per host halo
                    n_sub = nsub_per_host * n_host_halos
                    n_halos_tot = n_sub + n_host_halos

            logmu_obs: ndarray of shape (n_halos_tot, )
                base-10 log of mu=Msub/Mhost

            halo_indx: ndarray of shape (n_halos_tot, )
                index of the associated host halo
                for central halos: halo_indx = range(n_halos_tot)

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
    args = (ran_key, n_host_halos, z_min, z_max, lgmp_min, lgmp_max, sky_area_degsq)
    halopop = mcl.weighted_lc(
        *args, cosmo_params=cosmo_params, logmp_cutoff=logmp_cutoff
    )

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

    lc_data = passively_add_emlines_to_lc_data(ssp_data, lc_data)
    return lc_data


def passively_add_emlines_to_lc_data(ssp_data, lc_data):
    """Include precomputed emission line fluxes, if they are present in the ssp_data

    If ssp_data.emlines exists, the returned lc_data will have two additional fields:

        precomputed_ssp_lineflux_cgs_table : array, shape (n_lines, n_met, n_age)

        line_wave_table : array, shape (n_lines, )

    """
    if hasattr(ssp_data, "emlines"):

        precomputed_ssp_lineflux_cgs_table = jnp.array(
            [emline.line_flux for emline in ssp_data.emlines]
        )
        line_wave_table = jnp.array([emline.line_wave for emline in ssp_data.emlines])

        new_fields = ("precomputed_ssp_lineflux_cgs_table", "line_wave_table")
        new_vals = (precomputed_ssp_lineflux_cgs_table, line_wave_table)
        fields = (*LCData._fields, *new_fields)
        values = (*lc_data, *new_vals)
        lc_data = namedtuple("LCData", fields)(*values)

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
