"""
Empirical model fits for starforming and quiescent galaxies from
Mowla et al. 2019 ApJ...880...57 https://iopscience.iop.org/article/10.3847/1538-4357/ab290a/pdf
Kawinwanichakij et al. 2021 ApJ...921...38 https://iopscience.iop.org/article/10.3847/1538-4357/ac1f21/pdf
George, A. et al. 2024 MNRAS.528.4797 https://arxiv.org/pdf/2401.06842
"""

from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ...utils.utility_funcs import _sigmoid


@jjit
def _linear(x, ymin, slope):
    return ymin + slope * x


SigmoidParameters = namedtuple("SigmoidParameters", ("x0", "k", "ymin", "ymax"))

SizeParamsDisk = ["a_{}", "alpha_{}"]
NamesDisk = [p.format("disk") for p in SizeParamsDisk]
DiskSizeParameters = namedtuple("DiskSizeParameters", NamesDisk)

a_disk = SigmoidParameters(0.908, 2.42, 5.94, 3.82)
alpha_disk = SigmoidParameters(0.519, 56.4, 0.207, 0.181)

DISK_SIZE_PARAMETERS = DiskSizeParameters(a_disk, alpha_disk)

LinearParameters = namedtuple("LinearParameters", ("ymin", "slope"))

SizeParamsBulge = ["rp_{}", "alpha_{}", "beta_{}", "logmp_{}"]
NamesBulge = [p.format("bulge") for p in SizeParamsBulge]
BulgeSizeParameters = namedtuple("BulgeSizeParameters", NamesBulge)

rp_bulge = LinearParameters(2.04, -0.030)
alpha_bulge = LinearParameters(0.130, -0.0218)
beta_bulge = LinearParameters(0.71, 0.055)
logmp_bulge = LinearParameters(10.0, 0.40)

BULGE_SIZE_PARAMETERS = BulgeSizeParameters(
    rp_bulge, alpha_bulge, beta_bulge, logmp_bulge
)


@jjit
def median_r50_vs_mstar(mstar, a, alpha, m0=5e10):
    r50_med = a * jnp.power(mstar / m0, alpha)
    return r50_med


@jjit
def median_r50_vs_mstar2(mstar, rp, alpha, beta, logmp, delta=6):
    mp = jnp.power(10, logmp)
    term1 = rp * jnp.power(mstar / mp, alpha)
    term2 = 0.5 * jnp.power((1 + jnp.power(mstar / mp, delta)), (beta - alpha) / delta)
    r50_med = term1 * term2
    return r50_med


R50_MIN, R50_MAX = 0.1, 40.0
R50_SCATTER = 0.2


@jjit
def _get_parameter_zevolution_disk(redshift, parameters=DISK_SIZE_PARAMETERS):
    par_names = [par for par in parameters._fields]
    func_pars = [getattr(parameters, name) for name in par_names]
    evolved_parameters = [_sigmoid(redshift, *fpar) for fpar in func_pars]
    return evolved_parameters


@jjit
def _get_parameter_zevolution_bulge(redshift, parameters=BULGE_SIZE_PARAMETERS):
    par_names = [par for par in parameters._fields]
    func_pars = [getattr(parameters, name) for name in par_names]
    evolved_parameters = [_linear(redshift, *fpar) for fpar in func_pars]
    return evolved_parameters


@jjit
def _disk_median_r50(mstar, redshift):
    evolved_parameters = _get_parameter_zevolution_disk(redshift)
    r50 = median_r50_vs_mstar(mstar, *evolved_parameters)
    return r50


@jjit
def _bulge_median_r50(mstar, redshift):
    par_values = _get_parameter_zevolution_bulge(redshift)
    r50 = median_r50_vs_mstar2(mstar, *par_values)
    return r50


@jjit
def mc_r50_disk_size(mstar, redshift, ran_key):
    """
    mstar: array of length (Ngals), stellar masses of galaxies in units of Msun
    redshift: array of length Ngals, redshift of galaxies
    returns
    r50: array length (Ngals), size in kpc
    """
    logr50_med = jnp.log10(_disk_median_r50(mstar, redshift))
    logr50 = jran.normal(ran_key, shape=logr50_med.shape) * R50_SCATTER + logr50_med
    r50 = 10**logr50
    r50 = jnp.clip(r50, R50_MIN, R50_MAX)
    return r50


@jjit
def mc_r50_bulge_size(mstar, redshift, ran_key):
    """
    mstar: array of length (Ngals), stellar masses of galaxies in units of Msun
    redshift: array of length Ngals, redshift of galaxies
    returns
    r50: array length (Ngals), size in kpc
    """
    logr50_med = jnp.log10(_bulge_median_r50(mstar, redshift))
    logr50 = jran.normal(ran_key, shape=logr50_med.shape) * R50_SCATTER + logr50_med
    r50 = 10**logr50
    r50 = jnp.clip(r50, R50_MIN, R50_MAX)
    return r50
