""" """

import numpy as np
from dsps.cosmology import DEFAULT_COSMOLOGY

from .. import lc_utils as lcu


def test_spherical_shell_comoving_volume():
    z_grid = np.linspace(1, 2, 25)
    vol_shell_grid = lcu.spherical_shell_comoving_volume(z_grid, DEFAULT_COSMOLOGY)
    assert vol_shell_grid.shape == z_grid.shape
    assert np.all(np.isfinite(vol_shell_grid))
    assert np.all(vol_shell_grid > 0)
