""""""

from datetime import datetime

import h5py

from . import load_lc_cf

column_metadata = dict()
column_metadata["central"] = ("None", "0 for satellite, 1 for central")
column_metadata["core_tag"] = ("None", "ID of the simulated core")
column_metadata["ra"] = ("degrees", "right ascension")
column_metadata["dec"] = ("degrees", "declination")

column_metadata["early_index"] = ("None", "Diffmah parameter for halo mass assembly")
column_metadata["late_index"] = ("None", "Diffmah parameter for halo mass assembly")
column_metadata["logm0"] = ("None", "Diffmah parameter for halo mass assembly")
column_metadata["logtc"] = ("None", "Diffmah parameter for halo mass assembly")
column_metadata["t_peak"] = ("None", "Diffmah parameter for halo mass assembly")

column_metadata["has_diffmah_fit"] = ("None", "1 if core has a diffmah fit, 0 if not")
column_metadata["snapnum"] = ("None", "Timestep of the simulation")

column_metadata["lgmcrit"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["lgy_at_mcrit"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["indx_lo"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["indx_hi"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["tau_dep"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["lg_qt"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["lg_drop"] = ("None", "Diffstar parameter for galaxy SFH")
column_metadata["qlglgdt"] = ("None", "Diffstar parameter for galaxy SFH")
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
    import diffstarpop  # noqa
    import dsps  # noqa
    import jax  # noqa
    import numpy  # noqa

    import diffsky  # noqa

    version_info["diffmah"] = str(diffmah.__version__)
    version_info["diffsky"] = str(diffsky.__version__)
    version_info["diffstar"] = str(diffstar.__version__)
    version_info["diffstarpop"] = str(diffstarpop.__version__)
    version_info["dsps"] = str(dsps.__version__)
    version_info["jax"] = str(jax.__version__)
    version_info["numpy"] = str(numpy.__version__)

    return version_info
