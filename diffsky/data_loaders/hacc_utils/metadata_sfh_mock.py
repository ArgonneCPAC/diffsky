""""""

from datetime import datetime

import h5py

from . import load_lc_cf

column_metadata = dict()
column_metadata["central"] = ("None", "0 for satellite, 1 for central")
column_metadata["core_tag"] = ("None", "ID of the simulated core")
column_metadata["ra"] = ("degrees", "right ascension")
column_metadata["dec"] = ("degrees", "declination")

column_metadata["stepnum"] = (
    "None",
    "Timestep of the simulation. Varies between 0 and 499 for HACC gravity-only N-body simulations.",
)
column_metadata["lc_patch"] = ("None", "Lightcone patch number")

column_metadata["early_index"] = ("None", "Diffmah parameter for halo mass assembly")
column_metadata["late_index"] = ("None", "Diffmah parameter for halo mass assembly")
column_metadata["logm0"] = ("None", "Diffmah parameter for halo mass assembly")
column_metadata["logtc"] = ("None", "Diffmah parameter for halo mass assembly")
column_metadata["t_peak"] = ("None", "Diffmah parameter for halo mass assembly")

column_metadata["has_diffmah_fit"] = ("None", "1 if core has a diffmah fit, 0 if not")

column_metadata["lgmcrit"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["lgy_at_mcrit"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["indx_lo"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["indx_hi"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["lg_qt"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["qlglgdt"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["lg_drop"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["lg_rejuv"] = ("None", "Diffstar parameter for galaxy SFH")

column_metadata["logmp0"] = ("Msun (not Msun/h)", "Log10 of halo mass at z=0")
column_metadata["logmp_obs"] = ("Msun (not Msun/h)", "Log10 of halo mass at z")
column_metadata["logmp_obs_host"] = (
    "Msun (not Msun/h)",
    "Log10 of mass of parent halo at z",
)
column_metadata["logsm_obs"] = ("Msun (not Msun/h/h)", "Log10 of stellar mass at z")
column_metadata["logssfr_obs"] = ("1/yr ", "Log10(SFR/Mstar) at time of observation")

column_metadata["top_host_idx"] = ("None", "Index of host core")

column_metadata["x"] = ("Mpc/h", "Comoving coordinate of lightcone position")
column_metadata["y"] = ("Mpc/h", "Comoving coordinate of lightcone position")
column_metadata["z"] = ("Mpc/h", "Comoving coordinate of lightcone position")

column_metadata["x_host"] = ("Mpc/h", "Comoving coord of host halo lightcone position")
column_metadata["y_host"] = ("Mpc/h", "Comoving coord of host halo lightcone position")
column_metadata["z_host"] = ("Mpc/h", "Comoving coord of host halo lightcone position")

column_metadata["x_nfw"] = ("Mpc/h", "Comoving coord with NFW-repositioned satellites")
column_metadata["y_nfw"] = ("Mpc/h", "Comoving coord with NFW-repositioned satellites")
column_metadata["z_nfw"] = ("Mpc/h", "Comoving coord with NFW-repositioned satellites")

column_metadata["vx"] = ("km/s", "Comoving peculiar velocity in x-direction")
column_metadata["vy"] = ("km/s", "Comoving peculiar velocity in y-direction")
column_metadata["vz"] = ("km/s", "Comoving peculiar velocity in z-direction")

column_metadata["top_host_infall_fof_halo_eigS1X"] = (
    "Mpc/h",
    "x-component of first eigendirection of halo shape (unreduced intertia tensor)",
)
column_metadata["top_host_infall_fof_halo_eigS1Y"] = (
    "Mpc/h",
    "y-component of first eigendirection of halo shape (unreduced intertia tensor)",
)
column_metadata["top_host_infall_fof_halo_eigS1Z"] = (
    "Mpc/h",
    "z-component of first eigendirection of halo shape (unreduced intertia tensor)",
)

column_metadata["top_host_infall_fof_halo_eigS2X"] = (
    "Mpc/h",
    "x-component of second eigendirection of halo shape (unreduced intertia tensor)",
)
column_metadata["top_host_infall_fof_halo_eigS2Y"] = (
    "Mpc/h",
    "y-component of second eigendirection of halo shape (unreduced intertia tensor)",
)
column_metadata["top_host_infall_fof_halo_eigS2Z"] = (
    "Mpc/h",
    "z-component of second eigendirection of halo shape (unreduced intertia tensor)",
)

column_metadata["top_host_infall_fof_halo_eigS3X"] = (
    "Mpc/h",
    "x-component of third eigendirection of halo shape (unreduced intertia tensor)",
)
column_metadata["top_host_infall_fof_halo_eigS3Y"] = (
    "Mpc/h",
    "y-component of third eigendirection of halo shape (unreduced intertia tensor)",
)
column_metadata["top_host_infall_fof_halo_eigS3Z"] = (
    "Mpc/h",
    "z-component of third eigendirection of halo shape (unreduced intertia tensor)",
)


column_metadata["redshift_true"] = ("None", "True redshift")
column_metadata["ra_nfw"] = (
    "degrees",
    "Longitude in degrees. Based on NFW-repositioned satellites.",
)
column_metadata["dec_nfw"] = (
    "degrees",
    "Latitude in degrees. Based on NFW-repositioned satellites.",
)

column_metadata["uran_av"] = ("None", "Array for adding noise to dust parameter Av")
column_metadata["uran_delta"] = (
    "None",
    "Array for adding noise to dust parameter delta",
)
column_metadata["uran_funo"] = ("None", "Array for adding noise to dust parameter funo")

column_metadata["delta_scatter_ms"] = (
    "None",
    "Array for adding noise to SSP SED errors",
)
column_metadata["delta_scatter_q"] = (
    "None",
    "Array for adding noise to SSP SED errors",
)

column_metadata["bulge_to_total"] = ("None", "Bulge-to-total mass ratio")
column_metadata["fbulge_tcrit"] = ("None", "Bulge model parameter")
column_metadata["fbulge_early"] = ("None", "Bulge model parameter")
column_metadata["fbulge_late"] = ("None", "Bulge model parameter")

column_metadata["black_hole_mass"] = ("Msun", "Black hole mass")
column_metadata["black_hole_eddington_ratio"] = (
    "None",
    "dimensionless Eddington ratio",
)
column_metadata["black_hole_accretion_rate"] = (
    "Msun/yr",
    "Black hole mass accretion rate",
)

column_metadata["lgfburst"] = (
    "None",
    "Burstiness parameter. Specifies burst intensity.",
)
column_metadata["lgyr_max"] = (
    "None",
    "Burstiness parameter. Specifies burst duration.",
)
column_metadata["lgyr_peak"] = (
    "None",
    "Burstiness parameter. Specifies burst recency.",
)

column_metadata["lsst_u"] = ("None", "Apparent magnitude of composite galaxy")
column_metadata["lsst_g"] = ("None", "Apparent magnitude of composite galaxy")
column_metadata["lsst_r"] = ("None", "Apparent magnitude of composite galaxy")
column_metadata["lsst_i"] = ("None", "Apparent magnitude of composite galaxy")
column_metadata["lsst_z"] = ("None", "Apparent magnitude of composite galaxy")
column_metadata["lsst_y"] = ("None", "Apparent magnitude of composite galaxy")

column_metadata["lsst_u_bulge"] = ("None", "Apparent magnitude of bulge component")
column_metadata["lsst_g_bulge"] = ("None", "Apparent magnitude of bulge component")
column_metadata["lsst_r_bulge"] = ("None", "Apparent magnitude of bulge component")
column_metadata["lsst_i_bulge"] = ("None", "Apparent magnitude of bulge component")
column_metadata["lsst_z_bulge"] = ("None", "Apparent magnitude of bulge component")
column_metadata["lsst_y_bulge"] = ("None", "Apparent magnitude of bulge component")

column_metadata["lsst_u_disk"] = ("None", "Apparent magnitude of disk component")
column_metadata["lsst_g_disk"] = ("None", "Apparent magnitude of disk component")
column_metadata["lsst_r_disk"] = ("None", "Apparent magnitude of disk component")
column_metadata["lsst_i_disk"] = ("None", "Apparent magnitude of disk component")
column_metadata["lsst_z_disk"] = ("None", "Apparent magnitude of disk component")
column_metadata["lsst_y_disk"] = ("None", "Apparent magnitude of disk component")

column_metadata["lsst_u_knot"] = ("None", "Apparent magnitude of knot component")
column_metadata["lsst_g_knot"] = ("None", "Apparent magnitude of knot component")
column_metadata["lsst_r_knot"] = ("None", "Apparent magnitude of knot component")
column_metadata["lsst_i_knot"] = ("None", "Apparent magnitude of knot component")
column_metadata["lsst_z_knot"] = ("None", "Apparent magnitude of knot component")
column_metadata["lsst_y_knot"] = ("None", "Apparent magnitude of knot component")

column_metadata["fknot"] = (
    "None",
    "Fraction of disk mass contained in star-forming knots",
)

column_metadata["mc_sfh_type"] = (
    "None",
    "Boolean specifies the type of star formation history."
    "0 for quenched, 1 for smooth main sequence, 2 for bursty main sequence",
)


HEADER_COMMENT = """
This file contains diffsky galaxy data.
Each file stores mock galaxies in a thin redshift shell of a small patch of sky.
Each column of data includes metadata with units and comments about the column.
Contact: ahearin@anl.gov for questions.
"""


def append_metadata(fnout, sim_name):
    with h5py.File(fnout, "r+") as hdf_out:
        hdf_out.require_group("metadata")
        hdf_out.attrs["metadata/creation_date"] = str(datetime.now())

        hdf_out.attrs["metadata/header"] = HEADER_COMMENT

        sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim(sim_name)

        # Nbody simulation info
        nbody_group = hdf_out.require_group("metadata/nbody_info")
        nbody_group.attrs["sim_name"] = sim_name
        nbody_group.attrs["n_particles"] = sim_info.sim.np**3
        nbody_group.attrs["Lbox"] = sim_info.sim.rl / sim_info.cosmo_params.h

        mp = sim_info.sim.particle_mass / sim_info.cosmo_params.h  # Msun
        nbody_group.attrs["particle_mass"] = mp

        # Cosmology info
        cosmo_group = hdf_out.require_group("metadata/cosmology")
        cosmo_group.attrs["Om0"] = sim_info.sim.cosmo.Omega_m
        cosmo_group.attrs["w0"] = sim_info.sim.cosmo.w0
        cosmo_group.attrs["wa"] = sim_info.sim.cosmo.wa
        cosmo_group.attrs["h"] = sim_info.sim.cosmo.h
        cosmo_group.attrs["Ob0"] = sim_info.sim.cosmo.Omega_b
        cosmo_group.attrs["sigma8"] = sim_info.sim.cosmo.s8
        cosmo_group.attrs["ns"] = sim_info.sim.cosmo.ns

        # Software version info
        version_info_group = hdf_out.require_group("metadata/version_info")
        version_info = get_dependency_versions()
        for libname, version in version_info.items():
            version_info_group.attrs[libname] = version

        # Column metadata
        for key, val in column_metadata.items():
            key_out = "data/" + key
            assert key_out in hdf_out.keys(), f"{key_out} is missing from {fnout}"

            unit, description = val
            hdf_out[key_out].attrs["unit"] = unit
            hdf_out[key_out].attrs["description"] = description


def get_dependency_versions():
    version_info = dict()
    import diffmah  # noqa
    import diffstar  # noqa
    import dsps  # noqa
    import jax  # noqa
    import numpy  # noqa

    import diffsky  # noqa

    version_info["diffmah"] = str(diffmah.__version__)
    version_info["diffsky"] = str(diffsky.__version__)
    version_info["diffstar"] = str(diffstar.__version__)
    version_info["dsps"] = str(dsps.__version__)
    version_info["jax"] = str(jax.__version__)
    version_info["numpy"] = str(numpy.__version__)

    return version_info
