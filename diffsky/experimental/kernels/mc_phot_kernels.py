""""""

from . import (
    constants,
)
from . import dbk_kernels as dbk_kernels
from . import dbk_specphot_kernels as dbk_specphot_kernels
from . import (
    linelum_kernels,
    mc_randoms,
    phot_kernels,
    phot_kernels_merging,
)
from . import sed_kernels_in_situ as sed_kernels
from . import (
    specphot_kernels_merging,
)

# constants
LGMET_SCATTER = constants.LGMET_SCATTER


# kernels
_mc_phot_kern = phot_kernels._mc_phot_kern
_phot_kern = phot_kernels._phot_kern
_dbk_kern = dbk_kernels._dbk_kern  # noqa
_mc_specphot_kern = linelum_kernels._mc_specphot_kern
_specphot_kern = linelum_kernels._specphot_kern
_get_dbk_phot_from_dbk_weights = dbk_kernels._get_dbk_phot_from_dbk_weights
_sed_kern = sed_kernels._sed_kern
_get_dbk_linelum_decomposition = dbk_kernels._get_dbk_linelum_decomposition
_mc_phot_kern_merging = phot_kernels_merging._mc_phot_kern_merging
_phot_kern_merging = phot_kernels_merging._phot_kern_merging
_mc_specphot_kern_merging = specphot_kernels_merging._mc_specphot_kern_merging
_specphot_kern_merging = specphot_kernels_merging._specphot_kern_merging

_mc_dbk_phot_kern = dbk_specphot_kernels._mc_dbk_phot_kern
_mc_dbk_specphot_kern = dbk_specphot_kernels._mc_dbk_specphot_kern
interp_vmap2 = phot_kernels.interp_vmap2


# randoms
get_mc_phot_randoms = mc_randoms.get_mc_phot_randoms
get_mc_dbk_randoms = mc_randoms.get_mc_dbk_randoms

# namedtuple containers
PhotRandoms = mc_randoms.PhotRandoms
SpecKernResults = linelum_kernels.SpecKernResults
PhotKernResults = phot_kernels.PhotKernResults
DBKRandoms = mc_randoms.DBKRandoms
SEDKernResults = sed_kernels.SEDKernResults
SEDKernResults = sed_kernels.SEDKernResults
MCDBKPhotInfo = dbk_specphot_kernels.MCDBKPhotInfo
