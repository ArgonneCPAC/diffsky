""""""

from datetime import datetime

import h5py

from . import load_lc_cf

column_metadata = dict()
column_metadata["central"] = ("None", "0 for satellite, 1 for central")
column_metadata["core_tag"] = ("None", "ID of the simulated core")
column_metadata["ra"] = ("None", "right ascension")
column_metadata["dec"] = ("None", "declination")

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

column_metadata["x"] = ("Mpc", "Cartesian coordinate of lightcone position")
column_metadata["y"] = ("Mpc", "Cartesian coordinate of lightcone position")
column_metadata["z"] = ("Mpc", "Cartesian coordinate of lightcone position")

column_metadata["x_host"] = ("Mpc", "Cartesian coord of host halo lightcone position")
column_metadata["y_host"] = ("Mpc", "Cartesian coord of host halo lightcone position")
column_metadata["z_host"] = ("Mpc", "Cartesian coord of host halo lightcone position")

column_metadata["z_true"] = ("None", "True redshift")


HEADER_COMMENT = """
This file contains diffsky galaxy data.
Contact: ahearin@anl.gov for questions.
"""


def append_metadata(fnout, sim_name):
    with h5py.File(fnout, "r+") as hdf_out:
        hdf_out.attrs["creation_date"] = str(datetime.now())

        hdf_out.attrs["header"] = HEADER_COMMENT

        sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim(sim_name)

        # Nbody simulation info
        nbody_group = hdf_out.require_group("nbody_info")
        nbody_group.attrs["sim_name"] = sim_name
        nbody_group.attrs["n_particles"] = sim_info.sim.np**3
        nbody_group.attrs["Lbox"] = sim_info.sim.rl / sim_info.cosmo_params.h

        mp = sim_info.sim.particle_mass / sim_info.cosmo_params.h  # Msun
        nbody_group.attrs["particle_mass"] = mp

        # Cosmology info
        cosmo_group = hdf_out.require_group("cosmology")
        cosmo_group.attrs["Om0"] = sim_info.sim.cosmo.Omega_m
        cosmo_group.attrs["w0"] = sim_info.sim.cosmo.w0
        cosmo_group.attrs["wa"] = sim_info.sim.cosmo.wa
        cosmo_group.attrs["h"] = sim_info.sim.cosmo.n
        cosmo_group.attrs["Ob0"] = sim_info.sim.cosmo.Omega_b
        cosmo_group.attrs["sigma8"] = sim_info.sim.cosmo.s8
        cosmo_group.attrs["ns"] = sim_info.sim.cosmo.ns

        # Software version info
        version_info_group = hdf_out.require_group("version_info")
        version_info = get_dependency_versions()
        for libname, version in version_info.items():
            version_info_group.attrs[libname] = version

        # Column metadata
        for key, val in column_metadata.items():
            assert key in hdf_out.keys(), f"{key} is missing from {fnout}"

            units, comment = val
            hdf_out[key].attrs["units"] = units
            hdf_out[key].attrs["comment"] = comment


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
