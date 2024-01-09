"""
"""
from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import vmap

from .avpop import (
    DEFAULT_AVPOP_PARAMS,
    get_av_from_avpop_params,
    get_bounded_avpop_params,
    get_unbounded_avpop_params,
)
from .deltapop import (
    DEFAULT_DELTAPOP_PARAMS,
    get_bounded_deltapop_params,
    get_delta_from_deltapop_params,
    get_unbounded_deltapop_params,
)
from .funopop import (
    DEFAULT_FUNOPOP_PARAMS,
    calc_age_dep_funo_from_params_scalar,
    get_bounded_funopop_params,
    get_funo_from_funopop_params,
    get_unbounded_funopop_params,
)
from .tw_dust import DustParams, calc_dust_frac_trans

DEFAULT_DUSTPOP_PARAMS_PDICT = OrderedDict(
    avpop_params=DEFAULT_AVPOP_PARAMS,
    deltapop_params=DEFAULT_DELTAPOP_PARAMS,
    funopop_params=DEFAULT_FUNOPOP_PARAMS,
)

DustPopParams = namedtuple("DustPopParams", DEFAULT_DUSTPOP_PARAMS_PDICT.keys())
DEFAULT_DUSTPOP_PARAMS = DustPopParams(**DEFAULT_DUSTPOP_PARAMS_PDICT)

_DUSTPOP_UPNAMES = [
    key.replace("params", "u_params") for key in DEFAULT_DUSTPOP_PARAMS_PDICT.keys()
]
DustPopUParams = namedtuple("DustPopUParams", _DUSTPOP_UPNAMES)


@jjit
def get_bounded_dustpop_params(dustpop_u_params):
    avpop_u_params, deltapop_u_params, funopop_u_params = dustpop_u_params
    avpop_params = get_bounded_avpop_params(avpop_u_params)
    deltapop_params = get_bounded_deltapop_params(deltapop_u_params)
    funopop_params = get_bounded_funopop_params(funopop_u_params)
    diffburstpop_params = DustPopParams(avpop_params, deltapop_params, funopop_params)
    return diffburstpop_params


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
def calc_dust_ftrans_singlegal_singlewave_from_dustpop_params(
    dustpop_params, wave_aa, logsm, logssfr, lgfburst, ssp_lg_age_gyr
):
    """

    Parameters
    ----------
    wave_aa : float
        Wavelength in angstrom

    dustpop_params : namedtuple
        (avpop_params, deltapop_params, funopop_params)

    logsm : float
        Base-10 log of stellar mass in units of Msun

    logssfr : float
        Base-10 log of specific star formation rate sfr/Mstar in units of yr^-1

    lgfburst : float
        Base-10 log of burst fraction

    ssp_lg_age_gyr : ndarray, shape (n_age, )
        Base-10 log of stellar age in units of Gyr

    Returns
    -------
    frac_trans : float
        Fraction of light transmitted through dust at the input wavelength

    """
    av = get_av_from_avpop_params(dustpop_params.avpop_params, logsm, logssfr)
    delta = get_delta_from_deltapop_params(
        dustpop_params.deltapop_params, logsm, logssfr
    )
    funo = get_funo_from_funopop_params(
        dustpop_params.funopop_params, logsm, lgfburst, logssfr, ssp_lg_age_gyr
    )

    dust_params = DustParams(av, delta, funo)
    ftrans = calc_dust_frac_trans(wave_aa, dust_params)

    return ftrans


@jjit
def calc_dust_ftrans_singlegal_singlewave_singleage_from_dustpop_params(
    dustpop_params, wave_aa, logsm, logssfr, lgfburst, ssp_lg_age_gyr_scalar
):
    """

    Parameters
    ----------
    wave_aa : float
        Wavelength in angstrom

    dustpop_params : namedtuple
        (avpop_params, deltapop_params, funopop_params)

    logsm : float
        Base-10 log of stellar mass in units of Msun

    logssfr : float
        Base-10 log of specific star formation rate sfr/Mstar in units of yr^-1

    lgfburst : float
        Base-10 log of burst fraction

    ssp_lg_age_gyr_scalar : float
        Base-10 log of stellar age in units of Gyr

    Returns
    -------
    frac_trans : float
        Fraction of light transmitted through dust at the input wavelength

    """
    av = get_av_from_avpop_params(dustpop_params.avpop_params, logsm, logssfr)
    delta = get_delta_from_deltapop_params(
        dustpop_params.deltapop_params, logsm, logssfr
    )
    funo = calc_age_dep_funo_from_params_scalar(
        dustpop_params.funopop_params,
        logsm,
        lgfburst,
        logssfr,
        ssp_lg_age_gyr_scalar,
    )

    dust_params = DustParams(av, delta, funo)
    ftrans = calc_dust_frac_trans(wave_aa, dust_params)

    return ftrans


DEFAULT_DUSTPOP_U_PARAMS = DustPopUParams(
    *get_unbounded_dustpop_params(DEFAULT_DUSTPOP_PARAMS)
)

_A = (None, 0, None, None, None, None)
_B = [None, None, None, None, None, 0]

_f = jjit(
    vmap(
        calc_dust_ftrans_singlegal_singlewave_singleage_from_dustpop_params, in_axes=_A
    )
)
calc_dust_ftrans_singlegal_multiwave_from_dustpop_params = jjit(vmap(_f, in_axes=_B))
