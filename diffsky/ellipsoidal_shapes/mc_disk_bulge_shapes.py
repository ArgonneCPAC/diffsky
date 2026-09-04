""""""

from collections import namedtuple

from jax import numpy as jnp
from jax import random as jran
from jax import jit as jjit

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
    disk_params=disk_shapes.DEFAULT_DISK_PARAMS,
    bulge_params=bulge_shapes.DEFAULT_BULGE_PARAMS,
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

    disk_axis_ratios = disk_shapes.sample_disk_axis_ratios(
        disk_shape_key, n, disk_params=disk_params
    )
    bulge_axis_ratios = bulge_shapes.sample_bulge_axis_ratios(
        bulge_shape_key, n, bulge_params=bulge_params
    )

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
    # Draw random 3D orthonormal vectors A, B, C in the simulation frame
    # to be replaced by A, B, C = central_alignment/satellite_alignment in future
    A, B, C = epk.mc_orthonormal_vectors(ran_key, n)

    # Build the West-North-LoS coordinate system in the simulation frame
    obs_x, obs_y, obs_z = epk.observer_frame_axes_from_xyz(pos_x, pos_y, pos_z)

    # Use encapsulated function to compute the Ellipse2DParams
    a_disk = r50_disk
    b_disk = disk_axis_ratios.b_over_a * a_disk
    c_disk = disk_axis_ratios.c_over_a * a_disk

    a_bulge = r50_bulge
    b_bulge = bulge_axis_ratios.b_over_a * a_bulge
    c_bulge = bulge_axis_ratios.c_over_a * a_bulge

    disk_ellipse = epk.compute_ellipse2d_in_sim_frame(
        A, B, C, a_disk, b_disk, c_disk,
        obs_x, obs_y, obs_z,
        envelop=envelop,
        ellipticity_type=ellipticity_type,
    )

    bulge_ellipse = epk.compute_ellipse2d_in_sim_frame(
        A, B, C, a_bulge, b_bulge, c_bulge,
        obs_x, obs_y, obs_z,
        envelop=envelop,
        ellipticity_type=ellipticity_type,
    )

    # inject a random misalignment between the disk and bulge position angles
    psi_noise_rad = jnp.deg2rad(psi_noise_deg)
    delta_psi_rad = jran.normal(ran_key, shape=n) * psi_noise_rad
    bulge_ellipse = bulge_ellipse._replace(
        psi=bulge_ellipse.psi + delta_psi_rad
    )

    return disk_ellipse, bulge_ellipse
