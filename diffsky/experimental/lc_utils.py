"""Utility functions for lightcone calculations"""

from dsps.cosmology import flat_wcdm
from jax import grad
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

N_Z_GRID = 2_000
FULL_SKY_AREA = (4 * jnp.pi) * (180 / jnp.pi) ** 2


_Z = (0, None, None, None, None)
d_Rcom_dz_func = jjit(
    vmap(grad(flat_wcdm.comoving_distance_to_z, argnums=0), in_axes=_Z)
)


@jjit
def spherical_shell_comoving_volume(z_grid, cosmo_params):
    """Comoving volume of a spherical shell with width dR"""

    # Compute comoving distance to each grid point
    r_grid = flat_wcdm.comoving_distance(z_grid, *cosmo_params)

    # Compute dR = (dR/dz)*dz
    d_r_grid_dz = d_Rcom_dz_func(z_grid, *cosmo_params)
    d_z_grid = z_grid[1] - z_grid[0]
    d_r_grid = d_r_grid_dz * d_z_grid

    # vol_shell_grid = 4π*R*R*dR
    vol_shell_grid = 4 * jnp.pi * r_grid * r_grid * d_r_grid
    return vol_shell_grid
