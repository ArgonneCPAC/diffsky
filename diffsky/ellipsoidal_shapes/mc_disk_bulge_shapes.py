""""""

from jax import numpy as jnp
from jax import random as jran

from . import bulge_shapes, disk_shapes
from . import ellipse_proj_kernels as epk


def mc_disk_bulge_ellipsoids(ran_key, r50_disk, r50_bulge, psi_noise_deg=20.0):
    """Monte Carlo realization of disk/bulge axis ratios and orientations"""
    n = r50_disk.size

    ran_key, disk_shape_key, bulge_shape_key, los_key = jran.split(ran_key, 4)

    disk_axis_ratios = disk_shapes.sample_disk_axis_ratios(disk_shape_key, n)
    bulge_axis_ratios = bulge_shapes.sample_bulge_axis_ratios(bulge_shape_key, n)

    mu_ran, phi_ran = epk.mc_mu_phi(n, los_key)

    a_disk = r50_disk
    b_disk = disk_axis_ratios.b_over_a * a_disk
    c_disk = disk_axis_ratios.c_over_a * a_disk

    A_disk, B_disk, C_disk = epk._compute_2d_ellipse_params(
        a_disk, b_disk, c_disk, mu_ran, phi_ran
    )
    alpha_disk, beta_disk = epk._calculate_ellipse2d_axes(A_disk, B_disk, C_disk)
    ellipticity_disk = 1.0 - beta_disk / alpha_disk

    a_bulge = r50_bulge
    b_bulge = bulge_axis_ratios.b_over_a * a_bulge
    c_bulge = bulge_axis_ratios.c_over_a * a_bulge

    A_bulge, B_bulge, C_bulge = epk._compute_2d_ellipse_params(
        a_bulge, b_bulge, c_bulge, mu_ran, phi_ran
    )
    alpha_bulge, beta_bulge = epk._calculate_ellipse2d_axes(A_bulge, B_bulge, C_bulge)
    ellipticity_bulge = 1.0 - beta_bulge / alpha_bulge

    psi_disk_key, psi_bulge_key = jran.split(ran_key, 2)
    psi_disk = jran.uniform(psi_disk_key, minval=-jnp.pi, maxval=jnp.pi, shape=n)
    psi_noise_rad = jnp.deg2rad(psi_noise_deg)
    delta_psi_rad = jran.uniform(
        psi_bulge_key, minval=-psi_noise_rad, maxval=psi_noise_rad, shape=n
    )
    psi_bulge = psi_disk + delta_psi_rad

    xshift = psi_bulge + jnp.pi

    msk_shift_hi = xshift > 2 * jnp.pi
    xshifthi = jnp.where(msk_shift_hi, xshift - 2 * jnp.pi, xshift)

    msk_shift_lo = xshift < 0
    xshiftlo = jnp.where(msk_shift_lo, xshift + 2 * jnp.pi, xshifthi)

    psi_bulge = xshiftlo - jnp.pi

    return psi_bulge, psi_disk
