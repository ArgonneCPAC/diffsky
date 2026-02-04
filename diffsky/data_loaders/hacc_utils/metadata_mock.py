""""""

from datetime import datetime

import h5py

from . import load_lc_cf

try:
    from astropy import units as u
    from astropy.cosmology import units as cu

    u.add_enabled_units(cu)
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


HEADER_COMMENT = """
This file contains diffsky galaxy data.
Each file stores mock galaxies in a thin redshift shell of a small patch of sky.
Each column of data includes metadata with units and comments about the column.
Contact: ahearin@anl.gov for questions.
"""

COMPOSITE_MAG_MSG = "Apparent magnitude of composite galaxy"
LINEFLUX_MSG = "Emission line flux (continuum-subtracted)"
COMPONENT_MAG_MSG_PAT = "Apparent magnitude of {0} component"


def add_metadata_diffmah_columns(metadata):
    if not HAS_ASTROPY:
        raise ImportError("Must have astropy installed to attach units to metadata")
    metadata["early_index"] = (
        str(u.dimensionless_unscaled),
        "Diffmah parameter for halo mass assembly",
    )
    metadata["late_index"] = (
        str(u.dimensionless_unscaled),
        "Diffmah parameter for halo mass assembly",
    )
    metadata["logm0"] = (
        str(u.dimensionless_unscaled),
        "Diffmah parameter for halo mass assembly",
    )
    metadata["logtc"] = (
        str(u.dimensionless_unscaled),
        "Diffmah parameter for halo mass assembly",
    )
    metadata["t_peak"] = (
        str(u.dimensionless_unscaled),
        "Diffmah parameter for halo mass assembly",
    )

    metadata["has_diffmah_fit"] = (
        str(u.dimensionless_unscaled),
        "1 if core has a diffmah fit, 0 if not",
    )

    logMsun = u.dex(u.Msun)

    metadata["logmp0"] = (str(logMsun), "Log10 of halo mass at z=0")
    metadata["logmp_obs"] = (str(logMsun), "Log10 of halo mass at z")
    metadata["logmp_obs_host"] = (
        str(logMsun),
        "Log10 of mass of parent halo at z",
    )

    return metadata


def add_metadata_spspop_columns(metadata):

    # Randoms
    metadata["uran_av"] = (
        str(u.dimensionless_unscaled),
        "Array for adding noise to dust parameter Av",
    )
    metadata["uran_delta"] = (
        str(u.dimensionless_unscaled),
        "Array for adding noise to dust parameter delta",
    )
    metadata["uran_funo"] = (
        str(u.dimensionless_unscaled),
        "Array for adding noise to dust parameter funo",
    )
    metadata["uran_pburst"] = (
        str(u.dimensionless_unscaled),
        "Array for adding noise to burstiness",
    )

    metadata["delta_mag_ssp_scatter"] = (
        str(u.dimensionless_unscaled),
        "Array for adding noise to SSP SED errors",
    )

    # dust
    metadata["av"] = (
        str(u.dimensionless_unscaled),
        "Dust attenuation curve parameter",
    )
    metadata["delta"] = (
        str(u.dimensionless_unscaled),
        "Dust attenuation curve parameter",
    )
    metadata["funo"] = (
        str(u.dimensionless_unscaled),
        "Dust attenuation curve parameter",
    )

    # burstiness
    metadata["lgfburst"] = (
        str(u.dimensionless_unscaled),
        "Burstiness parameter. Specifies burst intensity.",
    )
    metadata["lgyr_max"] = (
        str(u.dimensionless_unscaled),
        "Burstiness parameter. Specifies burst duration.",
    )
    metadata["lgyr_peak"] = (
        str(u.dimensionless_unscaled),
        "Burstiness parameter. Specifies burst recency.",
    )

    return metadata


def add_metadata_diffstar_columns(metadata):

    metadata["lgmcrit"] = (
        str(u.dimensionless_unscaled),
        "Diffstar parameter for galaxy SFH",
    )
    metadata["lgy_at_mcrit"] = (
        str(u.dimensionless_unscaled),
        "Diffstar parameter for galaxy SFH",
    )
    metadata["indx_lo"] = (
        str(u.dimensionless_unscaled),
        "Diffstar parameter for galaxy SFH",
    )
    metadata["indx_hi"] = (
        str(u.dimensionless_unscaled),
        "Diffstar parameter for galaxy SFH",
    )
    metadata["lg_qt"] = (
        str(u.dimensionless_unscaled),
        "Diffstar parameter for galaxy SFH",
    )
    metadata["qlglgdt"] = (
        str(u.dimensionless_unscaled),
        "Diffstar parameter for galaxy SFH",
    )
    metadata["lg_drop"] = (
        str(u.dimensionless_unscaled),
        "Diffstar parameter for galaxy SFH",
    )
    metadata["lg_rejuv"] = (
        str(u.dimensionless_unscaled),
        "Diffstar parameter for galaxy SFH",
    )

    logMsun = u.dex(u.Msun)
    metadata["logsm_obs"] = (
        str(logMsun),
        "Log10 of stellar mass at z",
    )
    metadata["logssfr_obs"] = (
        str(u.dex(1 / u.yr)),
        "Log10(SFR/Mstar) at time of observation",
    )

    metadata["mc_sfh_type"] = (
        str(u.dimensionless_unscaled),
        "Boolean specifies the type of star formation history."
        "0 for quenched, 1 for smooth main sequence, 2 for bursty main sequence",
    )

    return metadata


def add_metadata_nfw_columns(metadata):
    unit_mpch = u.Mpc / cu.littleh

    metadata["x_nfw"] = (
        str(unit_mpch),
        "Comoving coord with NFW-repositioned satellites",
    )
    metadata["y_nfw"] = (
        str(unit_mpch),
        "Comoving coord with NFW-repositioned satellites",
    )
    metadata["z_nfw"] = (
        str(unit_mpch),
        "Comoving coord with NFW-repositioned satellites",
    )

    metadata["ra_nfw"] = (
        str(u.deg),
        "Longitude in degrees. Based on NFW-repositioned satellites.",
    )
    metadata["dec_nfw"] = (
        str(u.deg),
        "Latitude in degrees. Based on NFW-repositioned satellites.",
    )

    return metadata


def add_metadata_posvel_columns(metadata):
    unit_mpch = u.Mpc / cu.littleh

    metadata["ra"] = (str(u.deg), "right ascension")
    metadata["dec"] = (str(u.deg), "declination")

    metadata["x"] = (
        str(unit_mpch),
        "Comoving coordinate of lightcone position",
    )
    metadata["y"] = (
        str(unit_mpch),
        "Comoving coordinate of lightcone position",
    )
    metadata["z"] = (
        str(unit_mpch),
        "Comoving coordinate of lightcone position",
    )

    metadata["x_host"] = (
        str(unit_mpch),
        "Comoving coord of host halo lightcone position",
    )
    metadata["y_host"] = (
        str(unit_mpch),
        "Comoving coord of host halo lightcone position",
    )
    metadata["z_host"] = (
        str(unit_mpch),
        "Comoving coord of host halo lightcone position",
    )

    metadata["vpec"] = (
        str(u.km / u.s),
        "Comoving peculiar velocity along line-of-sight to galaxy",
    )
    metadata["vx"] = (
        str(u.km / u.s),
        "Comoving peculiar velocity in Cartesian x-direction",
    )
    metadata["vy"] = (
        str(u.km / u.s),
        "Comoving peculiar velocity in Cartesian y-direction",
    )
    metadata["vz"] = (
        str(u.km / u.s),
        "Comoving peculiar velocity in Cartesian z-direction",
    )
    metadata["msk_v0"] = (
        str(u.dimensionless_unscaled),
        "Boolean mask is True when the original core velocity on the lightcone was exactly zero and was overwritten",
    )
    return metadata


def add_metadata_inertia_tensor_columns(metadata):
    if not HAS_ASTROPY:
        raise ImportError("Must have astropy installed to attach units to metadata")

    unit_mpch = u.Mpc / cu.littleh

    metadata["top_host_infall_fof_halo_eigS1X"] = (
        str(unit_mpch),
        "x-component of first eigendirection of halo shape (unreduced inertia tensor)",
    )
    metadata["top_host_infall_fof_halo_eigS1Y"] = (
        str(unit_mpch),
        "y-component of first eigendirection of halo shape (unreduced inertia tensor)",
    )
    metadata["top_host_infall_fof_halo_eigS1Z"] = (
        str(unit_mpch),
        "z-component of first eigendirection of halo shape (unreduced inertia tensor)",
    )

    metadata["top_host_infall_fof_halo_eigS2X"] = (
        str(unit_mpch),
        "x-component of second eigendirection of halo shape (unreduced inertia tensor)",
    )
    metadata["top_host_infall_fof_halo_eigS2Y"] = (
        str(unit_mpch),
        "y-component of second eigendirection of halo shape (unreduced inertia tensor)",
    )
    metadata["top_host_infall_fof_halo_eigS2Z"] = (
        str(unit_mpch),
        "z-component of second eigendirection of halo shape (unreduced inertia tensor)",
    )

    metadata["top_host_infall_fof_halo_eigS3X"] = (
        str(unit_mpch),
        "x-component of third eigendirection of halo shape (unreduced inertia tensor)",
    )
    metadata["top_host_infall_fof_halo_eigS3Y"] = (
        str(unit_mpch),
        "y-component of third eigendirection of halo shape (unreduced inertia tensor)",
    )
    metadata["top_host_infall_fof_halo_eigS3Z"] = (
        str(unit_mpch),
        "z-component of third eigendirection of halo shape (unreduced inertia tensor)",
    )
    return metadata


def add_metadata_morphology_columns(metadata):
    metadata["bulge_to_total"] = (
        str(u.dimensionless_unscaled),
        "Bulge-to-total mass ratio",
    )
    metadata["fbulge_tcrit"] = (
        str(u.dimensionless_unscaled),
        "Bulge model parameter",
    )
    metadata["fbulge_early"] = (
        str(u.dimensionless_unscaled),
        "Bulge model parameter",
    )
    metadata["fbulge_late"] = (
        str(u.dimensionless_unscaled),
        "Bulge model parameter",
    )
    return metadata


def add_metadata_black_hole_columns(metadata):

    metadata["black_hole_mass"] = (str(u.Msun), "Black hole mass")
    metadata["black_hole_eddington_ratio"] = (
        str(u.dimensionless_unscaled),
        "dimensionless Eddington ratio",
    )
    metadata["black_hole_accretion_rate"] = (
        str(u.Msun / u.yr),
        "Black hole mass accretion rate",
    )
    return metadata


def add_metadata_sed_columns(metadata):
    if not HAS_ASTROPY:
        raise ImportError("Must have astropy installed to attach units to metadata")


def add_metadata_dbk_sed_columns(metadata):
    metadata["fknot"] = (
        str(u.dimensionless_unscaled),
        "Fraction of disk mass contained in star-forming knots",
    )
    return metadata


def add_metadata_dbk_morphology_columns(metadata):
    metadata["r50_disk"] = (str(u.kpc), "Half-mass radius of disk")
    metadata["r50_bulge"] = (str(u.kpc), "Half-mass radius of bulge")
    metadata["zscore_r50_disk"] = (
        str(u.dimensionless_unscaled),
        "Gaussian random used to add scatter to disk radius",
    )
    metadata["zscore_r50_bulge"] = (
        str(u.dimensionless_unscaled),
        "Gaussian random used to add scatter to bulge radius",
    )

    metadata["b_over_a_disk"] = (
        str(u.dimensionless_unscaled),
        "3d axis ratio b/a, where 0 < b/a < 1",
    )
    metadata["c_over_a_disk"] = (
        str(u.dimensionless_unscaled),
        "3d axis ratio c/a, where 0 < c/a < 1",
    )
    metadata["b_over_a_bulge"] = (
        str(u.dimensionless_unscaled),
        "3d axis ratio b/a, where 0 < b/a < 1",
    )
    metadata["c_over_a_bulge"] = (
        str(u.dimensionless_unscaled),
        "3d axis ratio c/a, where 0 < c/a < 1",
    )

    metadata["beta_disk"] = (
        str(u.kpc),
        "2d projected size of the semi-major axis of the disk",
    )
    metadata["alpha_disk"] = (
        str(u.kpc),
        "2d projected size of the semi-minor axis of the disk",
    )
    metadata["ellipticity_disk"] = (
        str(u.dimensionless_unscaled),
        "2d projected size of the semi-major axis of the disk",
    )
    metadata["psi_disk"] = (
        str(u.dimensionless_unscaled),
        "Angular coordinate of projected semi-major axis of the disk, where 0<ψ<2π",
    )

    metadata["beta_bulge"] = (
        str(u.kpc),
        "2d projected size of the semi-major axis of the bulge",
    )
    metadata["alpha_bulge"] = (
        str(u.kpc),
        "2d projected size of the semi-minor axis of the bulge",
    )
    metadata["ellipticity_bulge"] = (
        str(u.dimensionless_unscaled),
        "2d projected size of the semi-major axis of the bulge",
    )
    metadata["psi_bulge"] = (
        str(u.dimensionless_unscaled),
        "Angular coordinate of projected semi-major axis of the bulge, where 0<ψ<2π",
    )

    metadata["e_beta_x_disk"] = (
        str(u.dimensionless_unscaled),
        "x-coord of unit-normalized semi-major axis of the disk",
    )
    metadata["e_beta_y_disk"] = (
        str(u.dimensionless_unscaled),
        "y-coord of unit-normalized semi-major axis of the disk",
    )
    metadata["e_alpha_x_disk"] = (
        str(u.dimensionless_unscaled),
        "x-coord of unit-normalized semi-minor axis of the disk",
    )
    metadata["e_alpha_y_disk"] = (
        str(u.dimensionless_unscaled),
        "y-coord of unit-normalized semi-minor axis of the disk",
    )

    metadata["e_beta_x_bulge"] = (
        str(u.dimensionless_unscaled),
        "x-coord of unit-normalized semi-major axis of the bulge",
    )
    metadata["e_beta_y_bulge"] = (
        str(u.dimensionless_unscaled),
        "y-coord of unit-normalized semi-major axis of the bulge",
    )
    metadata["e_alpha_x_bulge"] = (
        str(u.dimensionless_unscaled),
        "x-coord of unit-normalized semi-minor axis of the bulge",
    )
    metadata["e_alpha_y_bulge"] = (
        str(u.dimensionless_unscaled),
        "y-coord of unit-normalized semi-minor axis of the bulge",
    )

    return metadata


def get_metadata_all_columns():
    if not HAS_ASTROPY:
        raise ImportError("Must have astropy installed to attach units to metadata")

    metadata_all_columns = dict()

    metadata_all_columns = add_metadata_lc_core_data_columns(metadata_all_columns)

    metadata_all_columns = add_metadata_posvel_columns(metadata_all_columns)
    metadata_all_columns = add_metadata_nfw_columns(metadata_all_columns)

    metadata_all_columns = add_metadata_inertia_tensor_columns(metadata_all_columns)

    metadata_all_columns = add_metadata_diffmah_columns(metadata_all_columns)
    metadata_all_columns = add_metadata_diffstar_columns(metadata_all_columns)

    metadata_all_columns = add_metadata_spspop_columns(metadata_all_columns)
    metadata_all_columns = add_metadata_morphology_columns(metadata_all_columns)

    metadata_all_columns = add_metadata_black_hole_columns(metadata_all_columns)

    metadata_all_columns = add_metadata_dbk_sed_columns(metadata_all_columns)
    metadata_all_columns = add_metadata_dbk_morphology_columns(metadata_all_columns)

    return metadata_all_columns


def add_metadata_lc_core_data_columns(metadata):
    metadata["central"] = (
        str(u.dimensionless_unscaled),
        "0 for satellite, 1 for central",
    )
    metadata["core_tag"] = (
        str(u.dimensionless_unscaled),
        "ID of the simulated core",
    )

    metadata["top_host_idx"] = (
        str(u.dimensionless_unscaled),
        "Index of host core",
    )
    metadata["redshift_true"] = (
        str(u.dimensionless_unscaled),
        "True redshift",
    )

    metadata["stepnum"] = (
        str(u.dimensionless_unscaled),
        "Timestep of the simulation. Varies between 0 and 499 for HACC gravity-only N-body simulations.",
    )
    metadata["lc_patch"] = (
        str(u.dimensionless_unscaled),
        "Lightcone patch number",
    )

    return metadata


def append_metadata(
    fnout,
    sim_name,
    mock_version_name,
    z_phot_table,
    filter_nicknames,
    lineflux_nicknames,
    *,
    exclude_colnames=[],
    no_dbk=False,
):
    try:
        from astropy import units as u
        from astropy.cosmology import units as cu

    except ImportError:
        raise ImportError("Must have astropy installed to attach units to metadata")

    u.add_enabled_units(cu)

    column_metadata = get_metadata_all_columns()

    for colname in exclude_colnames:
        column_metadata.pop(colname)

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

            hdf_out[key_out].attrs["unit"] = str(u.ABmag)
            hdf_out[key_out].attrs["description"] = COMPOSITE_MAG_MSG

            # Component magnitudes
            if no_dbk:
                pass  # skip dbk metadata
            else:
                for component in ("bulge", "disk", "knots"):
                    key_out = "data/" + "_".join((nickname, component))
                    assert (
                        key_out in hdf_out.keys()
                    ), f"{key_out} is missing from {fnout}"

                    msg = COMPONENT_MAG_MSG_PAT.format(component)
                    hdf_out[key_out].attrs["unit"] = str(u.ABmag)
                    hdf_out[key_out].attrs["description"] = msg

        # emission line fluxes
        for linename in lineflux_nicknames:

            key_out = "data/" + linename
            assert key_out in hdf_out.keys(), f"{key_out} is missing from {fnout}"

            hdf_out[key_out].attrs["unit"] = str(u.erg / u.s)
            hdf_out[key_out].attrs["description"] = LINEFLUX_MSG


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
