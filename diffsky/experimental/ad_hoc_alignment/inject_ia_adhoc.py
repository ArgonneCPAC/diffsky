"""
Inject IA into DiffSky Mock Catalog
Author: Jiachuan Xu
This script is a standalone script that injects IA into the DiffSky Mock Catalog ad hoc with native DiffSky format.
IA strategy:
- resolved central galaxies: align with the host halo shape
- unresolved central galaxies: random orientation
- satellite galaxies: radial alignment to the position vector from the central galaxy to the satellite galaxy

Native DiffSky format: HDF5 file with filename convention
    lc_cores-aaa.b.diffsky_gals.hdf5
where aaa is the timestep index and b is the lightcone patch index. Redshift decreases as stepnumber increases.
The HDF5 file strcture is flat table. Example structure is shown below:
    data
      |- stepnum
      |- ra
      |- dec
      |- ...
      |- central
"""
import jax.numpy as jnp
import jax.random as jran
from diffsky.experimental.ad_hoc_alignment.intrinsic_alignment import align_to_halo, align_radially
from diffsky.ellipsoidal_shapes import ellipse_proj_kernels as eproj
from diffsky.ellipsoidal_shapes import mc_disk_bulge_shapes as mcdbs


def alignment_worker(dataset, mu_central, mu_satellite, central_mask, synthetic_mask,
                     box_size_in_Mpc=jnp.inf, envelope=True, ellipticity_type=0, seed=514):
    """
    Worker function to align the galaxies in the dataset according to the
    specified alignment parameters.

    The alignment is done in the following way:
    - Central galaxies:
        - Resolved: align to the host halo shape (central_alignment)
        - Unresolved: random orientation
    - Satellite galaxies:
        - Resolved and unresolved: align towards halo center (satellite_alignment)

    All the galaxies have both disk and bulge components sharing the same
    alignment method.

    Parameters
    ----------
    dataset : HDF5 table
        The dataset containing the galaxy information.
    mu_central : float or jax.numpy.ndarray
        The alignment strength mu for central galaxies. If float, the same alignment strength
        is applied to all central galaxies. If array, it should have the same length as the
        number of galaxies, and each galaxy has individual alignment strength.
        Unresolved (synthetic) centrals are set to mu=0 in this function.
    mu_satellite : float or jax.numpy.ndarray
        The alignment strength mu for satellite galaxies. If float, the same alignment strength
        is applied to all satellite galaxies. If array, it should have the same length as the
        number of galaxies, and each galaxy has individual alignment strength.
    central_mask : array-like
        Boolean mask, 1 = central galaxies, 0 = satellite galaxies.
    synthetic_mask : array-like
        Boolean mask, 1 = synthetic galaxies, 0 = resolved galaxies.
    box_size_in_Mpc : float, optional
        The size of the simulation box in Mpc. Default is np.inf (no periodic boundary conditions).
    envelope : bool, optional
        Whether to use the envelope method for projection. Default is True.
    ellipticity_type : int, optional
        The type of ellipticity to use for the projection. Default is 0.
    seed : int, optional
        Random seed for reproducibility. Default is 514.
    Returns
    -------
    galaxy_axes : dict
        Dictionary containing the aligned galaxy axes for each galaxy.
    Ellipse2D_disk : mc_disk_bulge_shapes.Ellipse2DParams
        The projected 2D ellipse parameters for the disk component of each galaxy,
        with its inclination mu_inclination and 3D axis ratios.
    Ellipse2D_bulge : mc_disk_bulge_shapes.Ellipse2DParams
        The same for the bulge component of each galaxy.
    """
    key_central, key_satellite = jran.split(jran.PRNGKey(seed))
    # Get the axes in the observed RA/Dec frame for each galaxy, for shape projection.
    # x_prime = west, y_prime = north, z_prime = -LOS (toward observer)
    x_prime, y_prime, z_prime = eproj.observer_frame_axes_from_ra_dec(dataset["ra_nfw"], dataset["dec_nfw"])

    # Get the 3D axes size of disk and bulge for each galaxy
    disk_3D_sizes = jnp.stack(
        [
            dataset["r50_disk_3d"],
            dataset["r50_disk_3d"] * dataset["b_over_a_disk"],
            dataset["r50_disk_3d"] * dataset["c_over_a_disk"],
        ], axis=-1
    )
    bulge_3D_sizes = jnp.stack(
        [
            dataset["r50_bulge_3d"],
            dataset["r50_bulge_3d"] * dataset["b_over_a_bulge"],
            dataset["r50_bulge_3d"] * dataset["c_over_a_bulge"]
        ], axis=-1
    )

    # Get the 3D axes direction of central and satellite galaxies.
    # Every galaxy is run through both strategies at full length (rather than
    # boolean-mask subsetting, which has data-dependent shape and isn't
    # jit-compatible), and the unused branch is discarded below with
    # jnp.where. Unresolved (synthetic) centrals are folded into the same
    # central_alignment call via a per-galaxy mu of 0 (random orientation)
    # instead of a separate branch.
    mu_central_arr = jnp.where(synthetic_mask, 0.0, mu_central)
    assert_eigS3_is_major_axis(dataset)
    print("Central alignment ...")
    central_axes = central_alignment(
        dataset['top_host_infall_fof_halo_eigS3X'],
        dataset['top_host_infall_fof_halo_eigS3Y'],
        dataset['top_host_infall_fof_halo_eigS3Z'],
        mu_central_arr,
        key_central,
    )
    print("Satellite alignment ...")
    satellite_axes = satellite_alignment(
        dataset['x_host'], dataset['y_host'], dataset['z_host'],
        dataset['x_nfw'], dataset['y_nfw'], dataset['z_nfw'],
        mu_satellite,
        key_satellite,
        box_size_in_cMpc=box_size_in_Mpc,
    )
    # merge the axes of different types into a single array for each axis
    cmask = central_mask[:, None]
    gals_A = jnp.where(cmask, central_axes[0], satellite_axes[0])
    gals_B = jnp.where(cmask, central_axes[1], satellite_axes[1])
    gals_C = jnp.where(cmask, central_axes[2], satellite_axes[2])

    # Measure the galaxy shapes in RA/Dec frame (disk+bulge and central+satellite)
    Ellipse2D_disk = eproj.compute_ellipse2d_in_sim_frame(
        gals_A, gals_B, gals_C,
        disk_3D_sizes[:, 0],
        disk_3D_sizes[:, 1],
        disk_3D_sizes[:, 2],
        x_prime, y_prime,
        envelop=envelope,
        ellipticity_type=ellipticity_type,
    )

    Ellipse2D_bulge = eproj.compute_ellipse2d_in_sim_frame(
        gals_A, gals_B, gals_C,
        bulge_3D_sizes[:, 0],
        bulge_3D_sizes[:, 1],
        bulge_3D_sizes[:, 2],
        x_prime, y_prime,
        envelop=envelope,
        ellipticity_type=ellipticity_type,
    )

    # Return the same Ellipse2DParams as mc_disk_bulge_ellipsoids, including the
    # inclinations μ = cos(i), which change with the new orientations: the disk's
    # is between its normal (minor axis C) and the LoS, the bulge's between its
    # major axis (A) and the LoS
    Ellipse2D_disk = mcdbs.Ellipse2DParams(
        **Ellipse2D_disk._asdict(),
        mu_inclination=jnp.abs(jnp.sum(gals_C * z_prime, axis=-1)),
        b_over_a=jnp.asarray(dataset["b_over_a_disk"]),
        c_over_a=jnp.asarray(dataset["c_over_a_disk"]),
    )
    Ellipse2D_bulge = mcdbs.Ellipse2DParams(
        **Ellipse2D_bulge._asdict(),
        mu_inclination=jnp.abs(jnp.sum(gals_A * z_prime, axis=-1)),
        b_over_a=jnp.asarray(dataset["b_over_a_bulge"]),
        c_over_a=jnp.asarray(dataset["c_over_a_bulge"]),
    )

    return Ellipse2D_disk, Ellipse2D_bulge, {"gal_A": gals_A, "gal_B": gals_B, "gal_C": gals_C}


def assert_eigS3_is_major_axis(dataset):
    """ Check that eigS3 is the host halo's major axis for every galaxy.
    The halo shape eigenvectors are scaled by their axis lengths (units of Mpc),
    so the major axis is the one with the largest magnitude. Ties are allowed,
    e.g. unresolved halos whose three eigenvectors are all zero.
    Parameters
    ----------
    dataset : dict
        The galaxy catalog, with columns top_host_infall_fof_halo_eigS{1,2,3}{X,Y,Z}.
    """
    prefix = "top_host_infall_fof_halo_eig"
    norm_S1, norm_S2, norm_S3 = (
        jnp.sqrt(
            dataset[f"{prefix}S{i}X"] ** 2
            + dataset[f"{prefix}S{i}Y"] ** 2
            + dataset[f"{prefix}S{i}Z"] ** 2
        )
        for i in (1, 2, 3)
    )
    is_major = (norm_S3 >= norm_S1) & (norm_S3 >= norm_S2)
    n_bad = int(jnp.sum(~is_major))
    assert n_bad == 0, (
        f"eigS3 is not the largest halo eigenvector for {n_bad} of "
        f"{is_major.size} galaxies, so it cannot be used as the halo major axis"
    )


def central_alignment(gal_axisA_x, gal_axisA_y, gal_axisA_z, mu, key):
    """ Intrinsic alignment for central galaxies
    This function is designed to support multiple central galaxy alignment strategies,
    but currently only implements the "align to halo" strategy.
    Parameters
    ----------
    gal_axisA_x, gal_axisA_y, gal_axisA_z : array-like, shape (N,)
        The x, y, and z components of the host halo's major axis, for every
        galaxy (not just centrals) -- the caller selects which rows matter.
    mu : float or array-like, shape (N,)
        The alignment strength mu for central galaxies. Pass a per-galaxy
        array with 0.0 for unresolved/synthetic centrals to randomize them.
    key : jax.random.PRNGKey
        Random key for JAX random number generation.
    Returns
    -------
    gal_A : jax.numpy.ndarray
        The aligned galaxy axis A for central galaxies.
    gal_B : jax.numpy.ndarray
        The aligned galaxy axis B for central galaxies.
    gal_C : jax.numpy.ndarray
        The aligned galaxy axis C for central galaxies.
    """
    gal_A, gal_B, gal_C = align_to_halo(
        key, gal_axisA_x, gal_axisA_y, gal_axisA_z, mu, prim_gal_axis='A',
    )
    return gal_A, gal_B, gal_C


def satellite_alignment(halo_x, halo_y, halo_z, x, y, z, mu, key, box_size_in_cMpc=jnp.inf):
    """ Intrinsic alignment for satellite galaxies
    This function is designed to support multiple satellite galaxy alignment strategies,
    but currently only implements the "radial alignment" strategy.
    Parameters
    ----------
    halo_x, halo_y, halo_z : array-like, shape (N,)
        Host halo center position, for every galaxy (not just satellites) --
        the caller selects which rows matter.
    x, y, z : array-like, shape (N,)
        Galaxy position, for every galaxy.
    mu : float or array-like, shape (N,)
        The alignment strength mu for satellite galaxies.
    key : jax.random.PRNGKey
        Random key for JAX random number generation.
    box_size_in_cMpc : float, optional
        The size of the simulation box in comoving Mpc. Default is np.inf
    Returns
    -------
    gal_A : jax.numpy.ndarray
        The aligned galaxy axis A for satellite galaxies.
    gal_B : jax.numpy.ndarray
        The aligned galaxy axis B for satellite galaxies.
    gal_C : jax.numpy.ndarray
        The aligned galaxy axis C for satellite galaxies.
    """
    gal_A, gal_B, gal_C = align_radially(
        key, halo_x, halo_y, halo_z, x, y, z, mu,
        jnp.array([box_size_in_cMpc, box_size_in_cMpc, box_size_in_cMpc]),
        prim_gal_axis='A',
    )
    return gal_A, gal_B, gal_C


def main():
    import h5py
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Inject IA into DiffSky Mock Catalog")
    parser.add_argument("input_file", type=str, help="Input HDF5 file")
    parser.add_argument("output_file", type=str, help="Output HDF5 file")
    parser.add_argument("--mu_central", type=float, default=0.5, help="Alignment strength for central galaxies")
    parser.add_argument("--mu_satellite", type=float, default=0.5, help="Alignment strength for satellite galaxies")
    parser.add_argument("--seed", type=int, default=514, help="Random seed for reproducibility")
    args = parser.parse_args()

    with h5py.File(args.input_file, "r") as f:
        # "data" is a Group of one 1D Dataset per column, not a single
        # compound-dtype Dataset, so it can't be read with f["data"][:]
        dataset = {k: f["data"][k][:] for k in f["data"].keys()}
        central_mask = dataset["central"] == 1
        # A central galaxy is "unresolved" when its host halo shape wasn't
        # measured, which shows up as an exactly-zero eigenvector
        eig_norm = np.sqrt(
            dataset["top_host_infall_fof_halo_eigS3X"] ** 2
            + dataset["top_host_infall_fof_halo_eigS3Y"] ** 2
            + dataset["top_host_infall_fof_halo_eigS3Z"] ** 2
        )
        synthetic_mask = central_mask & (eig_norm == 0)

        # Calculate the aligned disk and bulge shapes
        # axis_disk is for debug and code comparison purpose, not saved.
        Ellipse2D_disk, Ellipse2D_bulge, axes_disk = alignment_worker(
            dataset,
            args.mu_central,
            args.mu_satellite,
            central_mask,
            synthetic_mask,
            seed=args.seed
        )

        # overwrite the original dataset with the new shapes
        # involved properties: ("alpha", "beta", "psi", "ellipticity", "r50",
        # "mu_inclination"); the 3D axis ratios are unchanged
        dataset["alpha_disk"] = Ellipse2D_disk.alpha
        dataset["beta_disk"] = Ellipse2D_disk.beta
        dataset["psi_disk"] = Ellipse2D_disk.psi
        dataset["ellipticity_disk"] = Ellipse2D_disk.ellipticity
        dataset["r50_disk_2d"] = Ellipse2D_disk.alpha * 0.765
        dataset["mu_inclination_disk"] = Ellipse2D_disk.mu_inclination
        dataset["alpha_bulge"] = Ellipse2D_bulge.alpha
        dataset["beta_bulge"] = Ellipse2D_bulge.beta
        dataset["psi_bulge"] = Ellipse2D_bulge.psi
        dataset["ellipticity_bulge"] = Ellipse2D_bulge.ellipticity
        dataset["r50_bulge_2d"] = Ellipse2D_bulge.alpha * 0.765
        dataset["mu_inclination_bulge"] = Ellipse2D_bulge.mu_inclination
        # columns for debug: the axes of the 3D galaxy ellipsoid after alignment
        # dataset["gal_Ax"] = axes_disk["gal_A"][:, 0]
        # dataset["gal_Ay"] = axes_disk["gal_A"][:, 1]
        # dataset["gal_Az"] = axes_disk["gal_A"][:, 2]
        # dataset["gal_Bx"] = axes_disk["gal_B"][:, 0]
        # dataset["gal_By"] = axes_disk["gal_B"][:, 1]
        # dataset["gal_Bz"] = axes_disk["gal_B"][:, 2]
        # dataset["gal_Cx"] = axes_disk["gal_C"][:, 0]
        # dataset["gal_Cy"] = axes_disk["gal_C"][:, 1]
        # dataset["gal_Cz"] = axes_disk["gal_C"][:, 2]

        with h5py.File(args.output_file, "w") as f_out:
            # carry over everything besides the per-galaxy columns unchanged,
            # e.g. the "metadata" group (cosmology, Lbox, software versions)
            f_out.attrs.update(f.attrs)
            for name in f.keys():
                if name != "data":
                    f.copy(f[name], f_out, name=name)

            grp = f_out.create_group("data")
            grp.attrs.update(f["data"].attrs)
            for key, val in dataset.items():
                grp.create_dataset(key, data=np.asarray(val))
                # keep the per-column "unit"/"description" attributes
                if key in f["data"]:
                    grp[key].attrs.update(f["data"][key].attrs)


if __name__ == "__main__":
    main()
