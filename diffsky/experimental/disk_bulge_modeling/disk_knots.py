"""
Add knot component
"""
from diffstar.utils import cumulative_mstar_formed
from dsps.constants import SFR_MIN
from dsps.sed.stellar_age_weights import calc_age_weights_from_sfh_table
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

FKNOT_MAX = 0.2


@jjit
def _disk_knot_kern(
    tarr, tobs, sfh, sfh_disk, fburst, fknot, age_weights_burst, ssp_lg_age_gyr
):
    sfh = jnp.where(sfh < SFR_MIN, SFR_MIN, sfh)
    sfh_disk = jnp.where(sfh_disk < SFR_MIN, SFR_MIN, sfh_disk)

    sfh_knot = fknot * sfh_disk
    sfh_diffuse_disk = sfh_disk * (1 - fknot)

    smh = cumulative_mstar_formed(tarr, sfh)
    smh_knot = cumulative_mstar_formed(tarr, sfh_knot)
    smh_diffuse_disk = cumulative_mstar_formed(tarr, sfh_diffuse_disk)

    age_weights_dd = calc_age_weights_from_sfh_table(
        tarr, sfh_diffuse_disk, ssp_lg_age_gyr, tobs
    )

    lgt_table = jnp.log10(tarr)
    mstar_tot = 10 ** jnp.interp(jnp.log10(tobs), lgt_table, jnp.log10(smh))
    mknot = 10 ** jnp.interp(jnp.log10(tobs), lgt_table, jnp.log10(smh_knot))
    mdd = 10 ** jnp.interp(jnp.log10(tobs), lgt_table, jnp.log10(smh_diffuse_disk))
    mburst = fburst * mstar_tot

    mburst_by_mknot = mburst / mknot
    burst_knot_age_weights = (
        mburst_by_mknot * age_weights_burst + (1 - mburst_by_mknot) * age_weights_dd
    )
    age_weights_knot = jnp.where(
        mburst_by_mknot > 1, age_weights_burst, burst_knot_age_weights
    )

    mburst_dd = jnp.where(mburst_by_mknot > 1, mburst - mknot, 0.0)

    mdd_tot = mdd + mburst_dd
    age_weights_dd = (mdd / mdd_tot) * age_weights_dd + (
        mburst_dd / mdd_tot
    ) * age_weights_burst

    return mstar_tot, mburst, mdd, mknot, age_weights_dd, age_weights_knot


_V = (None, 0, 0, 0, 0, 0, 0, None)
_disk_knot_vmap = jjit(vmap(_disk_knot_kern, in_axes=_V))
