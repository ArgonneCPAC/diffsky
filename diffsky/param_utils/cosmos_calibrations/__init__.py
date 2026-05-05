""""""

from .cosmos_params_260105 import cosmos_260105 as _cosmos_260105
from .cosmos_params_260120_UM import cosmos_260120_UM as _cosmos_260120_UM
from .cosmos_params_260210 import cosmos_260210 as _cosmos_260210
from .cosmos_params_260215 import cosmos_260215 as _cosmos_260215
from .cosmos_params_260305_Cmrg import cosmos_260305_Cmrg as _cosmos_260305_Cmrg
from .cosmos_params_260316 import cosmos_260316 as _cosmos_260316

COSMOS_PARAM_FITS = dict()

COSMOS_PARAM_FITS["cosmos_260105"] = _cosmos_260105
COSMOS_PARAM_FITS["cosmos_260120_UM"] = _cosmos_260120_UM
COSMOS_PARAM_FITS["cosmos_260210"] = _cosmos_260210
COSMOS_PARAM_FITS["cosmos_260215"] = _cosmos_260215

COSMOS_PARAM_FITS_MERGING = dict()
COSMOS_PARAM_FITS_MERGING["cosmos_260305_Cmrg"] = _cosmos_260305_Cmrg
COSMOS_PARAM_FITS_MERGING["cosmos_260316"] = _cosmos_260316
