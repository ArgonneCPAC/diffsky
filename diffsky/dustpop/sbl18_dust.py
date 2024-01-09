"""
"""
from collections import OrderedDict, namedtuple

from jax import jit as jjit

from .sbl18_dust_kernels import sbl18_dust_transmission

MICRON_PER_AA = 1 / 10_000
DEFAULT_UV_BUMP_AMP = 1.0

DEFAULT_DUST_PARAMS_PDICT = OrderedDict(av=0.5, delta=0.0, funo=0.05)
DustParams = namedtuple("DustParams", DEFAULT_DUST_PARAMS_PDICT.keys())
DEFAULT_DUST_PARAMS = DustParams(**DEFAULT_DUST_PARAMS_PDICT)

DEFAULT_DUST_PBOUNDS_PDICT = OrderedDict(
    av=(0.01, 10.0), delta=(-0.9, 0.9), funo=(0.001, 0.25)
)
DEFAULT_DUST_PBOUNDS = DustParams(**DEFAULT_DUST_PBOUNDS_PDICT)


@jjit
def calc_dust_frac_trans(wave_aa, dust_params):
    wave_micron = wave_aa * MICRON_PER_AA

    ftrans = sbl18_dust_transmission(
        wave_micron,
        dust_params.av,
        dust_params.delta,
        dust_params.funo,
        DEFAULT_UV_BUMP_AMP,
    )
    return ftrans
