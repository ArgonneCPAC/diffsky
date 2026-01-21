""""""

from .cosmos_params_260105 import cosmos_260105 as _cosmos_260105
from .cosmos_params_260120_UM import cosmos_260120_UM as _cosmos_260120_UM

COSMOS_PARAM_FITS = dict()

COSMOS_PARAM_FITS["cosmos_260105"] = _cosmos_260105
COSMOS_PARAM_FITS["cosmos_260120_UM"] = _cosmos_260120_UM
