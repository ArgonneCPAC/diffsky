""""""

import jax.numpy as jnp
from dsps.constants import L_SUN_CGS
from dsps.utils import _sigmoid
from jax import jit as jjit
from jax import vmap


@jjit
def _fake_lineflux_kern(ssp_lgmet, ssp_lg_age_gyr):
    ylo = _sigmoid(ssp_lgmet, -3.5, 10.0, 25.0, 15.0)
    lineflux_per_msun = _sigmoid(ssp_lg_age_gyr + 9.0, 6.8, 10.0, ylo, 0.0)
    return lineflux_per_msun


_A = (0, None)
_B = (None, 0)
_fake_lineflux_table = jjit(vmap(vmap(_fake_lineflux_kern, in_axes=_B), in_axes=_A))


@jjit
def fake_lineflux_table_cgs(ssp_lgmet, ssp_lg_age_gyr):
    """Load dummy lineflux data for unit testing

    Parameters
    ----------
    ssp_lgmet : array, shape (n_met, )

    ssp_lg_age_gyr : array, shape (n_age, )

    Returns
    -------
    lineflux_table : array, shape (n_met, n_age)
        Line luminosity in erg/s/Msun

    """
    return _fake_lineflux_table(ssp_lgmet, ssp_lg_age_gyr) * L_SUN_CGS


@jjit
def get_ssp_linelum(emlines_wave_aa, ssp_data):
    ssp_emline_wave = jnp.array(ssp_data.ssp_emline_wave)

    ssp_linewave_idx = []
    for emline_wave_aa in emlines_wave_aa:
        idx = jnp.argmin(jnp.abs(ssp_emline_wave - emline_wave_aa))
        ssp_linewave_idx.append(idx)
    ssp_linewave_idx = jnp.array(ssp_linewave_idx)
    ssp_linelum = ssp_data.ssp_emline_luminosity[:, :, ssp_linewave_idx]

    return ssp_linelum, ssp_linewave_idx
