""""""

from collections import namedtuple

from jax import numpy as jnp
from jax import random as jran

from . import bulge_shapes, disk_shapes
from . import ellipse_proj_kernels as epk

Ellipse2DParams = namedtuple(
    "Ellipse2DParams",
    ("alpha", "beta", "psi", "ellipticity", "e_alpha", "e_beta", "A", "B", "C"),
)


def mc_disk_bulge_ellipsoids(
    ran_key,
    r50_disk,
    r50_bulge,
    pos_x,
    pos_y,
    pos_z,
    psi_noise_deg=20.0,
    envelop=True,
    ellipticity_type=0,
):
    """Monte Carlo realization of disk/bulge axis ratios and orientations

    Note: When we describe the line-of-sight direction with (mu, phi), the coordinate system of the u-v plane
    perpendicular to the LoS is free up to a rotation about the LoS direction. The new implementation of shape
    projection requires a new degree-of-freedom (omega) to fix this arbitrariness, making the u-axis of the
    projected u-v plane aligned with the west direction in the sky (decreasing RA), and v-axis aligned with
    the north direction in the sky.

    Therefore, the coordinates of the galaxy are required to properly project the galaxy shape onto
    the west-north plane. We assume the standard mapping between RA/Dec and the comoving x/y/z coordinates
    in the simulation frame:
        (x, y, z) = (-cos δ * cos α, -cos δ * sin α, sin δ) * chi,
    where δ is declination and α is right ascension, chi is the looking back comoving distance. This means that
    the -x direction is where RA = 0 and Dec = 0, and +z direction is the north celestial pole.

    Under the special case of no intrinsic alignment, the (mu, phi, omega) expressed in the body frame are
    uniformly random. The rigorous projection can be skipped under this scenario. However, we still follow the
    full projection in the current implementation, and leave the IA implementation for future.

    Note: omega does not change the projected 2D ellipticity, it only changes the position angle of the 2D ellipse.
    """
    n = r50_disk.size

    ran_key, disk_shape_key, bulge_shape_key, los_key = jran.split(ran_key, 4)

    disk_axis_ratios = disk_shapes.sample_disk_axis_ratios(disk_shape_key, n)
    bulge_axis_ratios = bulge_shapes.sample_bulge_axis_ratios(bulge_shape_key, n)

    """
    The projection works in the following way: 
    We first define the direction of the 3D galaxy ellipsoid eigenaxes A, B, C in the simulation frame. This 
    could be a uniform random orientation when no intrinsic alignment is modeled, or non-uniform random
    orientation with a preferential direction when intrinsic alignment is modeled.
    Then, based on the coordinates of the galaxy, we define the west-north-LoS coordinate system in the simulation
    frame. The west-north-LoS axes are then transformed into the ellipsoid body frame, and projected onto the
    west-north plane. The new omega degree-of-freedom is introduced here, in addition to the other two dofs, 
    to define a ZXZ Euler rotation such that the body frame is rotated to the west-north-LoS frame. Then, 
    the 3D ellipsoid projection can be done with the three angles. 

    Whitout intrinsic alignment, the (mu, phi, omega) expressed in the body frame are uniformly random. In
    this special case, the rigorous projection can be skipped by just drawing random (mu, phi, omega) and
    project onto the simulation x-y frame. However, when intrinsic alignment is modeled, the (mu, phi, omega)
    expressed in the body frame are no longer uniformly random, and the rigorous projection is required.
    """
    ### (mu_ran, phi_ran, omega_ran) defines the 3D galaxy ellipsoid eigenaxes A, B, C in the simulation frame.
    ### The ellipsoid body frame x'-y'-z' is defined such that A=x', B=y', C=z'
    ### mu = cos(theta) where theta is the polar angle of C
    ### phi is the azimuthal angle of C
    ### omega is the angle between A and the pivot axis (z x z')
    ### Without intrinsic alignment, the (mu, phi, omega) are drawn from uniform random distribution.
    mu_ran, phi_ran, omega_ran = epk.mc_mu_phi_omega(n, los_key)

    ### Build the 3D vectors of A, B, and C in the simulation frame
    R = epk._get_eulerxzx_matrix_from_angles(
        phi_ran + jnp.pi / 2.0, jnp.arccos(mu_ran), omega_ran
    )
    A, B, C = R[:, :, 0], R[:, :, 1], R[:, :, 2]

    ### Build the West-North-LoS coordinate system in the simulation frame
    NCP = jnp.array([0, 0, 1])
    obs_z = jnp.stack([pos_x, pos_y, pos_z], axis=-1)
    obs_z = (
        -1 * obs_z / jnp.linalg.norm(obs_z, axis=-1, keepdims=True)
    )  # LoS points to the observer

    # obs_x = NCP x obs_z vanishes exactly at the celestial poles (obs_z || NCP), where
    # west/east is undefined anyway; fall back to a fixed default there. The vector fed
    # to jnp.linalg.norm is swapped in *before* the norm is taken (not the norm's output
    # afterward), so norm() is never evaluated at the zero vector -- its gradient (v/|v|)
    # would be a 0/0 NaN there, and jnp.where can't rescue a NaN already computed on the
    # discarded branch. This keeps the whole thing jit-safe with well-defined gradients.
    obs_x_raw = jnp.cross(NCP, obs_z)
    at_pole = jnp.linalg.norm(obs_x_raw, axis=-1, keepdims=True) < 1e-10
    obs_x_default = jnp.array([1.0, 0.0, 0.0])
    obs_x_safe = jnp.where(at_pole, obs_x_default, obs_x_raw)
    obs_x = obs_x_safe / jnp.linalg.norm(obs_x_safe, axis=-1, keepdims=True)

    obs_y = jnp.cross(obs_z, obs_x)

    ### Transform the West-North-LoS coordinate system into the ellipsoid body frame
    u = epk._transform_axes_to_frame(obs_x, A, B, C)
    v = epk._transform_axes_to_frame(obs_y, A, B, C)

    ### Get the projection angles in the ellipsoid body frame,
    ### which are the Euler angles of the ZXZ rotation from the body frame
    ### to the west-north-los frame
    z1, x2, z3 = epk._get_eulerzxz_angle_from_basis(u, v)
    mu_proj = jnp.cos(x2)
    phi_proj = z1 - jnp.pi / 2.0
    omega_proj = z3

    a_disk = r50_disk
    b_disk = disk_axis_ratios.b_over_a * a_disk
    c_disk = disk_axis_ratios.c_over_a * a_disk

    A_disk, B_disk, C_disk = epk._compute_2d_ellipse_params(
        a_disk,
        b_disk,
        c_disk,
        mu_proj,
        phi_proj,
        omega_proj,
        envelop=envelop,
    )
    alpha_disk, beta_disk = epk._calculate_ellipse2d_axes(A_disk, B_disk, C_disk)
    q_disk = beta_disk / alpha_disk
    if ellipticity_type == 0:
        ellipticity_disk = 1.0 - q_disk
    elif ellipticity_type == 1:
        ellipticity_disk = (1.0 - q_disk) / (1.0 + q_disk)
    elif ellipticity_type == 2:
        ellipticity_disk = (1.0 - q_disk**2) / (1.0 + q_disk**2)
    elif ellipticity_type == 3:
        ellipticity_disk = -jnp.log(q_disk)
    else:
        raise ValueError(
            f"Invalid ellipticity_type: {ellipticity_type}. Must be 0, 1, 2, or 3."
        )

    a_bulge = r50_bulge
    b_bulge = bulge_axis_ratios.b_over_a * a_bulge
    c_bulge = bulge_axis_ratios.c_over_a * a_bulge

    A_bulge, B_bulge, C_bulge = epk._compute_2d_ellipse_params(
        a_bulge, b_bulge, c_bulge, mu_proj, phi_proj, omega_proj, envelop=envelop
    )
    alpha_bulge, beta_bulge = epk._calculate_ellipse2d_axes(A_bulge, B_bulge, C_bulge)
    q_bulge = beta_bulge / alpha_bulge
    if ellipticity_type == 0:
        ellipticity_bulge = 1.0 - q_bulge
    elif ellipticity_type == 1:
        ellipticity_bulge = (1.0 - q_bulge) / (1.0 + q_bulge)
    elif ellipticity_type == 2:
        ellipticity_bulge = (1.0 - q_bulge**2) / (1.0 + q_bulge**2)
    elif ellipticity_type == 3:
        ellipticity_bulge = -jnp.log(q_bulge)
    else:
        raise ValueError(
            f"Invalid ellipticity_type: {ellipticity_type}. Must be 0, 1, 2, or 3."
        )

    # How to deal with bulge-disk misalignment?
    # psi_disk_key, psi_bulge_key = jran.split(ran_key, 2)
    # psi_disk = jran.uniform(psi_disk_key, minval=-jnp.pi, maxval=jnp.pi, shape=n)
    # psi_noise_rad = jnp.deg2rad(psi_noise_deg)
    # delta_psi_rad = jran.normal(psi_bulge_key, shape=n) * psi_noise_rad
    # psi_bulge = psi_disk + delta_psi_rad
    psi_noise_rad = jnp.deg2rad(psi_noise_deg)
    delta_psi_rad = jran.normal(ran_key, shape=n) * psi_noise_rad
    psi_disk = epk._calculate_ellipse2d_psi(A_disk, B_disk, C_disk)
    psi_bulge = epk._calculate_ellipse2d_psi(A_bulge, B_bulge, C_bulge) + delta_psi_rad

    e_alpha_disk, e_beta_disk = epk._get_xy_coords_of_projected_semi_axes(psi_disk)
    e_alpha_bulge, e_beta_bulge = epk._get_xy_coords_of_projected_semi_axes(psi_bulge)

    disk_ellipse = Ellipse2DParams(
        alpha_disk,
        beta_disk,
        psi_disk,
        ellipticity_disk,
        e_alpha_disk,
        e_beta_disk,
        A_disk,
        B_disk,
        C_disk,
    )

    bulge_ellipse = Ellipse2DParams(
        alpha_bulge,
        beta_bulge,
        psi_bulge,
        ellipticity_bulge,
        e_alpha_bulge,
        e_beta_bulge,
        A_bulge,
        B_bulge,
        C_bulge,
    )

    return disk_ellipse, bulge_ellipse
