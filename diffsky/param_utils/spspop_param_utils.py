# flake8: noqa: E402
""" """

from jax import config

config.update("jax_enable_x64", True)

from collections import namedtuple

from jax import jit as jjit

from ..burstpop.diffqburstpop_mono import DiffburstPopParams, DiffburstPopUParams
from ..burstpop.fburstpop_mono import (
    DEFAULT_FBURSTPOP_PARAMS,
    ZEROBURST_FBURSTPOP_PARAMS,
    get_bounded_fburstpop_params,
    get_unbounded_fburstpop_params,
)
from ..burstpop.freqburst_mono import (
    DEFAULT_FREQBURST_PARAMS,
    ZEROBURST_FREQBURST_PARAMS,
    get_bounded_freqburst_params,
    get_unbounded_freqburst_params,
)
from ..burstpop.tburstpop import (
    DEFAULT_TBURSTPOP_PARAMS,
    get_bounded_tburstpop_params,
    get_unbounded_tburstpop_params,
)
from ..dustpop.avpop_mono import (
    DEFAULT_AVPOP_PARAMS,
    get_bounded_avpop_params,
    get_unbounded_avpop_params,
)
from ..dustpop.deltapop import (
    DEFAULT_DELTAPOP_PARAMS,
    get_bounded_deltapop_params,
    get_unbounded_deltapop_params,
)
from ..dustpop.funopop_ssfr import (
    DEFAULT_FUNOPOP_PARAMS,
    get_bounded_funopop_params,
    get_unbounded_funopop_params,
)
from ..dustpop.tw_dustpop_new import DustPopParams, DustPopUParams

DEFAULT_DIFFBURSTPOP_PARAMS = DiffburstPopParams(
    DEFAULT_FREQBURST_PARAMS, DEFAULT_FBURSTPOP_PARAMS, DEFAULT_TBURSTPOP_PARAMS
)
ZERO_DIFFBURSTPOP_PARAMS = DiffburstPopParams(
    ZEROBURST_FREQBURST_PARAMS, ZEROBURST_FBURSTPOP_PARAMS, DEFAULT_TBURSTPOP_PARAMS
)

DEFAULT_DUSTPOP_PARAMS = DustPopParams(
    DEFAULT_AVPOP_PARAMS, DEFAULT_DELTAPOP_PARAMS, DEFAULT_FUNOPOP_PARAMS
)

SPSPopParams = namedtuple("SPSPopParams", ["burstpop_params", "dustpop_params"])
DEFAULT_SPSPOP_PARAMS = SPSPopParams(
    DEFAULT_DIFFBURSTPOP_PARAMS, DEFAULT_DUSTPOP_PARAMS
)
SPSPopUParams = namedtuple("SPSPopUParams", ["u_burstpop_params", "u_dustpop_params"])


@jjit
def get_bounded_diffburstpop_params(u_params):
    u_freqburst_params, u_fburstpop_params, u_tburstpop_params = u_params
    bounded_freqburst_params = get_bounded_freqburst_params(u_freqburst_params)
    bounded_tburstpop_params = get_bounded_tburstpop_params(u_tburstpop_params)
    bounded_fburstpop_params = get_bounded_fburstpop_params(u_fburstpop_params)
    diffburstpop_params = DiffburstPopParams(
        bounded_freqburst_params, bounded_fburstpop_params, bounded_tburstpop_params
    )
    return diffburstpop_params


@jjit
def get_unbounded_diffburstpop_params(params):
    freqburst_params, fburstpop_params, tburstpop_params = params
    unbounded_freqburst_params = get_unbounded_freqburst_params(freqburst_params)
    unbounded_fburstpop_params = get_unbounded_fburstpop_params(fburstpop_params)
    unbounded_tburstpop_params = get_unbounded_tburstpop_params(tburstpop_params)
    diffburstpop_u_params = DiffburstPopUParams(
        unbounded_freqburst_params,
        unbounded_fburstpop_params,
        unbounded_tburstpop_params,
    )
    return diffburstpop_u_params


@jjit
def get_bounded_dustpop_params(dustpop_u_params):
    avpop_u_params, deltapop_u_params, funopop_u_params = dustpop_u_params
    avpop_params = get_bounded_avpop_params(avpop_u_params)
    deltapop_params = get_bounded_deltapop_params(deltapop_u_params)
    funopop_params = get_bounded_funopop_params(funopop_u_params)
    dustpop_params = DustPopParams(avpop_params, deltapop_params, funopop_params)
    return dustpop_params


@jjit
def get_unbounded_dustpop_params(dustpop_params):
    avpop_params, deltapop_params, funopop_params = dustpop_params
    avpop_u_params = get_unbounded_avpop_params(avpop_params)
    deltapop_u_params = get_unbounded_deltapop_params(deltapop_params)
    funopop_u_params = get_unbounded_funopop_params(funopop_params)
    dustpop_u_params = DustPopUParams(
        avpop_u_params, deltapop_u_params, funopop_u_params
    )
    return dustpop_u_params


@jjit
def get_bounded_spspop_params_tw_dust(spspop_u_params):
    burstpop_params = get_bounded_diffburstpop_params(spspop_u_params.u_burstpop_params)
    dustpop_params = get_bounded_dustpop_params(spspop_u_params.u_dustpop_params)
    return SPSPopParams(burstpop_params, dustpop_params)


@jjit
def get_unbounded_spspop_params_tw_dust(spspop_params):
    burstpop_params = get_unbounded_diffburstpop_params(spspop_params.burstpop_params)
    dustpop_params = get_unbounded_dustpop_params(spspop_params.dustpop_params)
    return SPSPopUParams(burstpop_params, dustpop_params)


DEFAULT_SPSPOP_U_PARAMS = get_unbounded_spspop_params_tw_dust(DEFAULT_SPSPOP_PARAMS)
DEFAULT_DUSTPOP_U_PARAMS = get_unbounded_dustpop_params(DEFAULT_DUSTPOP_PARAMS)
DEFAULT_DIFFBURSTPOP_U_PARAMS = get_unbounded_diffburstpop_params(
    DEFAULT_DIFFBURSTPOP_PARAMS
)
ZERO_DIFFBURSTPOP_U_PARAMS = get_unbounded_diffburstpop_params(ZERO_DIFFBURSTPOP_PARAMS)
