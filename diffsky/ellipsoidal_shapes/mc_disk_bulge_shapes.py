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


def mc_disk_bulge_ellipsoids(ran_key, r50_disk, r50_bulge, psi_noise_deg=20.0, envelop=True, ellipticity_type=0):
    """Monte Carlo realization of disk/bulge axis ratios and orientations"""
    n = r50_disk.size

    ran_key, disk_shape_key, bulge_shape_key, los_key = jran.split(ran_key, 4)

    disk_axis_ratios = disk_shapes.sample_disk_axis_ratios(disk_shape_key, n)
    bulge_axis_ratios = bulge_shapes.sample_bulge_axis_ratios(bulge_shape_key, n)

    mu_ran, phi_ran, omega_ran = epk.mc_mu_phi_omega(n, los_key)

    a_disk = r50_disk
    b_disk = disk_axis_ratios.b_over_a * a_disk
    c_disk = disk_axis_ratios.c_over_a * a_disk

    A_disk, B_disk, C_disk = epk._compute_2d_ellipse_params(
        a_disk, b_disk, c_disk, mu_ran, phi_ran, omega_ran, envelop=envelop,
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
        raise ValueError(f"Invalid ellipticity_type: {ellipticity_type}. Must be 0, 1, 2, or 3.")

    a_bulge = r50_bulge
    b_bulge = bulge_axis_ratios.b_over_a * a_bulge
    c_bulge = bulge_axis_ratios.c_over_a * a_bulge

    A_bulge, B_bulge, C_bulge = epk._compute_2d_ellipse_params(
        a_bulge, b_bulge, c_bulge, mu_ran, phi_ran, omega_ran, envelop=envelop
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
        raise ValueError(f"Invalid ellipticity_type: {ellipticity_type}. Must be 0, 1, 2, or 3.")

    # How to deal with bulge-disk misalignment? 
    # psi_disk_key, psi_bulge_key = jran.split(ran_key, 2)
    # psi_disk = jran.uniform(psi_disk_key, minval=-jnp.pi, maxval=jnp.pi, shape=n)
    # psi_noise_rad = jnp.deg2rad(psi_noise_deg)
    # delta_psi_rad = jran.normal(psi_bulge_key, shape=n) * psi_noise_rad
    # psi_bulge = psi_disk + delta_psi_rad
    psi_disk = epk._calculate_ellipse2d_psi(A_disk, B_disk, C_disk)
    psi_bulge = epk._calculate_ellipse2d_psi(A_bulge, B_bulge, C_bulge)

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
