""" """

from jax import jit as jjit
from jax import nn
from jax import numpy as jnp

from ..utils import _inverse_sigmoid
from . import avpop_mono, deltapop, funopop_ssfr, tw_dust


@jjit
def calc_ftrans_singlegal_singlewave_from_dustpop_params(
    dustpop_params,
    wave_aa,
    logsm,
    logssfr,
    redshift,
    ssp_lg_age_gyr,
    random_draw_av,
    random_draw_delta,
    random_draw_funo,
    scatter_params,
):

    av = avpop_mono.get_av_from_avpop_params_singlegal(
        dustpop_params.avpop_params, logsm, logssfr, redshift, ssp_lg_age_gyr
    )
    delta = deltapop.get_delta_from_deltapop_params(
        dustpop_params.deltapop_params, logsm, logssfr
    )
    funo = funopop_ssfr.get_funo_from_funopop_params(
        dustpop_params.funopop_params, logssfr
    )
    dust_params = tw_dust.DustParams(av, delta, funo)
    ftrans = tw_dust.calc_dust_frac_trans(wave_aa, dust_params)
    noisy_dust_params = get_noisy_dust_params(
        dust_params, random_draw_av, random_draw_delta, random_draw_funo, scatter_params
    )
    noisy_ftrans = tw_dust.calc_dust_frac_trans(wave_aa, noisy_dust_params)

    return ftrans, noisy_ftrans, dust_params, noisy_dust_params


@jjit
def get_noisy_dust_params(
    dust_params, random_draw_av, random_draw_delta, random_draw_funo, scatter_params
):
    suav = jnp.log(jnp.exp(dust_params.av) - 1)
    noisy_suav = _inverse_sigmoid(
        random_draw_av, suav, scatter_params.av_scatter, 0.0, 1.0
    )
    noisy_av = nn.softplus(noisy_suav)

    udelta = deltapop._get_unbounded_deltapop_param(
        dust_params.delta, deltapop.DELTAPOP_BOUNDS
    )
    noisy_udelta = _inverse_sigmoid(
        random_draw_delta, udelta, scatter_params.delta_scatter, 0.0, 1.0
    )
    noisy_delta = deltapop._get_bounded_deltapop_param(
        noisy_udelta, deltapop.DELTAPOP_BOUNDS
    )

    ufuno = funopop_ssfr._get_u_p_from_p_scalar(
        dust_params.funo, funopop_ssfr.FUNO_BOUNDS
    )
    noisy_ufuno = _inverse_sigmoid(
        random_draw_funo, ufuno, scatter_params.funo_scatter, 0.0, 1.0
    )
    noisy_funo = funopop_ssfr._get_p_from_u_p_scalar(
        noisy_ufuno, funopop_ssfr.FUNO_BOUNDS
    )

    noisy_dust_params = tw_dust.DustParams(noisy_av, noisy_delta, noisy_funo)
    return noisy_dust_params
