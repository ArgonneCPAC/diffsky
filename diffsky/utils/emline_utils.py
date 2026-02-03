""""""

from dsps.utils import _sigmoid
from jax import jit as jjit
from jax import vmap

L_SUN_CGS = 3.828e33  # Lsun in erg/s


@jjit
def _fake_lineflux_kern(ssp_lgmet, ssp_lg_age_gyr):
    ylo = _sigmoid(ssp_lgmet, -3.5, 10.0, 25.0, 15.0)
    lineflux_per_msun = _sigmoid(ssp_lg_age_gyr + 9.0, 6.8, 10.0, ylo, 0.0)
    return lineflux_per_msun


_A = (0, None)
_B = (None, 0)
_fake_lineflux_table = jjit(vmap(vmap(_fake_lineflux_kern, in_axes=_B), in_axes=_A))


@jjit
def fake_lineflux_table(ssp_lgmet, ssp_lg_age_gyr):
    """Load dummy lineflux data for unit testing

    Parameters
    ----------
    ssp_lgmet : array, shape (n_met, )

    ssp_lg_age_gyr : array, shape (n_age, )

    Returns
    -------
    lineflux_table : array, shape (n_met, n_age)
        Units of Lsun/Msun

    """
    return _fake_lineflux_table(ssp_lgmet, ssp_lg_age_gyr)
