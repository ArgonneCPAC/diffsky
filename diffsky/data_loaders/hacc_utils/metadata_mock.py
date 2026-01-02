""""""

from datetime import datetime

import h5py

from . import load_lc_cf

HEADER_COMMENT = """
This file contains diffsky galaxy data.
Each file stores mock galaxies in a thin redshift shell of a small patch of sky.
Each column of data includes metadata with units and comments about the column.
Contact: ahearin@anl.gov for questions.
"""

COMPOSITE_MAG_MSG = "Apparent magnitude of composite galaxy"
COMPONENT_MAG_MSG_PAT = "Apparent magnitude of {0} component"


def get_column_metadata(column_names=None):
    metadata_all_columns = get_metadata_all_columns()
    if column_names is None:
        column_names = list(metadata_all_columns.keys())
    column_metadata = {
        key: value for key, value in metadata_all_columns.items() if key in column_names
    }
    return column_metadata


def get_metadata_all_columns():

    metadata_all_columns = dict()
    metadata_all_columns["central"] = ("None", "0 for satellite, 1 for central")
    metadata_all_columns["core_tag"] = ("None", "ID of the simulated core")
    metadata_all_columns["ra"] = ("degrees", "right ascension")
    metadata_all_columns["dec"] = ("degrees", "declination")

    metadata_all_columns["stepnum"] = (
        "None",
        "Timestep of the simulation. Varies between 0 and 499 for HACC gravity-only N-body simulations.",
    )
    metadata_all_columns["lc_patch"] = ("None", "Lightcone patch number")

    metadata_all_columns["early_index"] = (
        "None",
        "Diffmah parameter for halo mass assembly",
    )
    metadata_all_columns["late_index"] = (
        "None",
        "Diffmah parameter for halo mass assembly",
    )
    metadata_all_columns["logm0"] = ("None", "Diffmah parameter for halo mass assembly")
    metadata_all_columns["logtc"] = ("None", "Diffmah parameter for halo mass assembly")
    metadata_all_columns["t_peak"] = (
        "None",
        "Diffmah parameter for halo mass assembly",
    )

    metadata_all_columns["has_diffmah_fit"] = (
        "None",
        "1 if core has a diffmah fit, 0 if not",
    )

    metadata_all_columns["lgmcrit"] = ("None", "Diffstar parameter for galaxy SFH")
    metadata_all_columns["lgy_at_mcrit"] = ("None", "Diffstar parameter for galaxy SFH")
    metadata_all_columns["indx_lo"] = ("None", "Diffstar parameter for galaxy SFH")
    metadata_all_columns["indx_hi"] = ("None", "Diffstar parameter for galaxy SFH")
    metadata_all_columns["lg_qt"] = ("None", "Diffstar parameter for galaxy SFH")
    metadata_all_columns["qlglgdt"] = ("None", "Diffstar parameter for galaxy SFH")
    metadata_all_columns["lg_drop"] = ("None", "Diffstar parameter for galaxy SFH")
    metadata_all_columns["lg_rejuv"] = ("None", "Diffstar parameter for galaxy SFH")

    metadata_all_columns["logmp0"] = ("Msun (not Msun/h)", "Log10 of halo mass at z=0")
    metadata_all_columns["logmp_obs"] = ("Msun (not Msun/h)", "Log10 of halo mass at z")
    metadata_all_columns["logmp_obs_host"] = (
        "Msun (not Msun/h)",
        "Log10 of mass of parent halo at z",
    )
    metadata_all_columns["logsm_obs"] = (
        "Msun (not Msun/h/h)",
        "Log10 of stellar mass at z",
    )
    metadata_all_columns["logssfr_obs"] = (
        "1/yr ",
        "Log10(SFR/Mstar) at time of observation",
    )

    metadata_all_columns["top_host_idx"] = ("None", "Index of host core")

    metadata_all_columns["x"] = ("Mpc/h", "Comoving coordinate of lightcone position")
    metadata_all_columns["y"] = ("Mpc/h", "Comoving coordinate of lightcone position")
    metadata_all_columns["z"] = ("Mpc/h", "Comoving coordinate of lightcone position")

    metadata_all_columns["x_host"] = (
        "Mpc/h",
        "Comoving coord of host halo lightcone position",
    )
    metadata_all_columns["y_host"] = (
        "Mpc/h",
        "Comoving coord of host halo lightcone position",
    )
    metadata_all_columns["z_host"] = (
        "Mpc/h",
        "Comoving coord of host halo lightcone position",
    )

    metadata_all_columns["x_nfw"] = (
        "Mpc/h",
        "Comoving coord with NFW-repositioned satellites",
    )
    metadata_all_columns["y_nfw"] = (
        "Mpc/h",
        "Comoving coord with NFW-repositioned satellites",
    )
    metadata_all_columns["z_nfw"] = (
        "Mpc/h",
        "Comoving coord with NFW-repositioned satellites",
    )

    metadata_all_columns["vx"] = ("km/s", "Comoving peculiar velocity in x-direction")
    metadata_all_columns["vy"] = ("km/s", "Comoving peculiar velocity in y-direction")
    metadata_all_columns["vz"] = ("km/s", "Comoving peculiar velocity in z-direction")

    metadata_all_columns["top_host_infall_fof_halo_eigS1X"] = (
        "Mpc/h",
        "x-component of first eigendirection of halo shape (unreduced intertia tensor)",
    )
    metadata_all_columns["top_host_infall_fof_halo_eigS1Y"] = (
        "Mpc/h",
        "y-component of first eigendirection of halo shape (unreduced intertia tensor)",
    )
    metadata_all_columns["top_host_infall_fof_halo_eigS1Z"] = (
        "Mpc/h",
        "z-component of first eigendirection of halo shape (unreduced intertia tensor)",
    )

    metadata_all_columns["top_host_infall_fof_halo_eigS2X"] = (
        "Mpc/h",
        "x-component of second eigendirection of halo shape (unreduced intertia tensor)",
    )
    metadata_all_columns["top_host_infall_fof_halo_eigS2Y"] = (
        "Mpc/h",
        "y-component of second eigendirection of halo shape (unreduced intertia tensor)",
    )
    metadata_all_columns["top_host_infall_fof_halo_eigS2Z"] = (
        "Mpc/h",
        "z-component of second eigendirection of halo shape (unreduced intertia tensor)",
    )

    metadata_all_columns["top_host_infall_fof_halo_eigS3X"] = (
        "Mpc/h",
        "x-component of third eigendirection of halo shape (unreduced intertia tensor)",
    )
    metadata_all_columns["top_host_infall_fof_halo_eigS3Y"] = (
        "Mpc/h",
        "y-component of third eigendirection of halo shape (unreduced intertia tensor)",
    )
    metadata_all_columns["top_host_infall_fof_halo_eigS3Z"] = (
        "Mpc/h",
        "z-component of third eigendirection of halo shape (unreduced intertia tensor)",
    )

    metadata_all_columns["redshift_true"] = ("None", "True redshift")
    metadata_all_columns["ra_nfw"] = (
        "degrees",
        "Longitude in degrees. Based on NFW-repositioned satellites.",
    )
    metadata_all_columns["dec_nfw"] = (
        "degrees",
        "Latitude in degrees. Based on NFW-repositioned satellites.",
    )

    metadata_all_columns["uran_av"] = (
        "None",
        "Array for adding noise to dust parameter Av",
    )
    metadata_all_columns["uran_delta"] = (
        "None",
        "Array for adding noise to dust parameter delta",
    )
    metadata_all_columns["uran_funo"] = (
        "None",
        "Array for adding noise to dust parameter funo",
    )
    metadata_all_columns["uran_pburst"] = (
        "None",
        "Array for adding noise to burstiness",
    )

    metadata_all_columns["delta_mag_ssp_scatter"] = (
        "None",
        "Array for adding noise to SSP SED errors",
    )

    metadata_all_columns["av"] = ("None", "Dust attenuation curve parameter")
    metadata_all_columns["delta"] = ("None", "Dust attenuation curve parameter")
    metadata_all_columns["funo"] = ("None", "Dust attenuation curve parameter")

    metadata_all_columns["bulge_to_total"] = ("None", "Bulge-to-total mass ratio")
    metadata_all_columns["fbulge_tcrit"] = ("None", "Bulge model parameter")
    metadata_all_columns["fbulge_early"] = ("None", "Bulge model parameter")
    metadata_all_columns["fbulge_late"] = ("None", "Bulge model parameter")

    metadata_all_columns["black_hole_mass"] = ("Msun", "Black hole mass")
    metadata_all_columns["black_hole_eddington_ratio"] = (
        "None",
        "dimensionless Eddington ratio",
    )
    metadata_all_columns["black_hole_accretion_rate"] = (
        "Msun/yr",
        "Black hole mass accretion rate",
    )

    metadata_all_columns["lgfburst"] = (
        "None",
        "Burstiness parameter. Specifies burst intensity.",
    )
    metadata_all_columns["lgyr_max"] = (
        "None",
        "Burstiness parameter. Specifies burst duration.",
    )
    metadata_all_columns["lgyr_peak"] = (
        "None",
        "Burstiness parameter. Specifies burst recency.",
    )

    metadata_all_columns["fknot"] = (
        "None",
        "Fraction of disk mass contained in star-forming knots",
    )

    metadata_all_columns["mc_sfh_type"] = (
        "None",
        "Boolean specifies the type of star formation history."
        "0 for quenched, 1 for smooth main sequence, 2 for bursty main sequence",
    )

    metadata_all_columns["r50_disk"] = ("kpc", "Half-mass radius of disk")
    metadata_all_columns["r50_bulge"] = ("kpc", "Half-mass radius of bulge")
    metadata_all_columns["zscore_r50_disk"] = (
        "None",
        "Gaussian random used to add scatter to disk radius",
    )
    metadata_all_columns["zscore_r50_bulge"] = (
        "None",
        "Gaussian random used to add scatter to bulge radius",
    )

    metadata_all_columns["b_over_a_disk"] = (
        "None",
        "3d axis ratio b/a, where 0 < b/a < 1",
    )
    metadata_all_columns["c_over_a_disk"] = (
        "None",
        "3d axis ratio c/a, where 0 < c/a < 1",
    )
    metadata_all_columns["b_over_a_bulge"] = (
        "None",
        "3d axis ratio b/a, where 0 < b/a < 1",
    )
    metadata_all_columns["c_over_a_bulge"] = (
        "None",
        "3d axis ratio c/a, where 0 < c/a < 1",
    )

    metadata_all_columns["beta_disk"] = (
        "kpc",
        "2d projected size of the semi-major axis of the disk",
    )
    metadata_all_columns["alpha_disk"] = (
        "kpc",
        "2d projected size of the semi-minor axis of the disk",
    )
    metadata_all_columns["ellipticity_disk"] = (
        "None",
        "2d projected size of the semi-major axis of the disk",
    )
    metadata_all_columns["psi_disk"] = (
        "None",
        "Angular coordinate of projected semi-major axis of the disk, where 0<ψ<2π",
    )

    metadata_all_columns["beta_bulge"] = (
        "kpc",
        "2d projected size of the semi-major axis of the bulge",
    )
    metadata_all_columns["alpha_bulge"] = (
        "kpc",
        "2d projected size of the semi-minor axis of the bulge",
    )
    metadata_all_columns["ellipticity_bulge"] = (
        "None",
        "2d projected size of the semi-major axis of the bulge",
    )
    metadata_all_columns["psi_bulge"] = (
        "None",
        "Angular coordinate of projected semi-major axis of the bulge, where 0<ψ<2π",
    )

    metadata_all_columns["e_beta_x_disk"] = (
        "None",
        "x-coord of unit-normalized semi-major axis of the disk",
    )
    metadata_all_columns["e_beta_y_disk"] = (
        "None",
        "y-coord of unit-normalized semi-major axis of the disk",
    )
    metadata_all_columns["e_alpha_x_disk"] = (
        "None",
        "x-coord of unit-normalized semi-minor axis of the disk",
    )
    metadata_all_columns["e_alpha_y_disk"] = (
        "None",
        "y-coord of unit-normalized semi-minor axis of the disk",
    )

    metadata_all_columns["e_beta_x_bulge"] = (
        "None",
        "x-coord of unit-normalized semi-major axis of the bulge",
    )
    metadata_all_columns["e_beta_y_bulge"] = (
        "None",
        "y-coord of unit-normalized semi-major axis of the bulge",
    )
    metadata_all_columns["e_alpha_x_bulge"] = (
        "None",
        "x-coord of unit-normalized semi-minor axis of the bulge",
    )
    metadata_all_columns["e_alpha_y_bulge"] = (
        "None",
        "y-coord of unit-normalized semi-minor axis of the bulge",
    )

    return metadata_all_columns


def append_metadata(fnout, sim_name, mock_version_name, z_phot_table, filter_nicknames):
    column_metadata = get_column_metadata()

    with h5py.File(fnout, "r+") as hdf_out:

        metadata_group = hdf_out.require_group("metadata")

        metadata_group["z_phot_table"] = z_phot_table

        metadata_group.attrs["creation_date"] = str(datetime.now())
        metadata_group.attrs["mock_version_name"] = mock_version_name
        metadata_group.attrs["README"] = HEADER_COMMENT

        sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim(sim_name)

        # Nbody simulation info
        nbody_group = metadata_group.require_group("nbody_info")
        nbody_group.attrs["sim_name"] = sim_name
        nbody_group.attrs["n_particles"] = sim_info.sim.np**3
        nbody_group.attrs["Lbox"] = sim_info.sim.rl / sim_info.cosmo_params.h

        mp = sim_info.sim.particle_mass / sim_info.cosmo_params.h  # Msun
        nbody_group.attrs["particle_mass"] = mp

        # Cosmology info
        cosmo_group = metadata_group.require_group("cosmology")
        cosmo_group.attrs["Om0"] = sim_info.sim.cosmo.Omega_m
        cosmo_group.attrs["w0"] = sim_info.sim.cosmo.w0
        cosmo_group.attrs["wa"] = sim_info.sim.cosmo.wa
        cosmo_group.attrs["h"] = sim_info.sim.cosmo.h
        cosmo_group.attrs["Ob0"] = sim_info.sim.cosmo.Omega_b
        cosmo_group.attrs["sigma8"] = sim_info.sim.cosmo.s8
        cosmo_group.attrs["ns"] = sim_info.sim.cosmo.ns

        # Software version info
        software_version_info_group = metadata_group.require_group(
            "software_version_info"
        )
        software_version_info = get_dependency_versions()
        for libname, version in software_version_info.items():
            software_version_info_group.attrs[libname] = version

        # Column metadata
        for key, val in column_metadata.items():
            key_out = "data/" + key
            assert key_out in hdf_out.keys(), f"{key_out} is missing from {fnout}"

            unit, description = val
            hdf_out[key_out].attrs["unit"] = unit
            hdf_out[key_out].attrs["description"] = description

        # Filter magnitudes
        for nickname in filter_nicknames:

            # Composite magnitudes
            key_out = "data/" + nickname
            assert key_out in hdf_out.keys(), f"{key_out} is missing from {fnout}"

            hdf_out[key_out].attrs["unit"] = "None"
            hdf_out[key_out].attrs["description"] = COMPOSITE_MAG_MSG

            # Component magnitudes
            for component in ("bulge", "disk", "knots"):
                key_out = "data/" + "_".join((nickname, component))
                assert key_out in hdf_out.keys(), f"{key_out} is missing from {fnout}"

                msg = COMPONENT_MAG_MSG_PAT.format(component)
                hdf_out[key_out].attrs["unit"] = "None"
                hdf_out[key_out].attrs["description"] = msg


def get_dependency_versions():
    software_version_info = dict()
    import diffmah  # noqa
    import diffstar  # noqa
    import dsps  # noqa
    import jax  # noqa
    import numpy  # noqa

    import diffsky  # noqa

    software_version_info["diffmah"] = str(diffmah.__version__)
    software_version_info["diffsky"] = str(diffsky.__version__)
    software_version_info["diffstar"] = str(diffstar.__version__)
    software_version_info["dsps"] = str(dsps.__version__)
    software_version_info["jax"] = str(jax.__version__)
    software_version_info["numpy"] = str(numpy.__version__)

    return software_version_info
