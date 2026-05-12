"""
- Compute shamnet SMF loss/gradients using the MC halo generator. Three cases:
    1) no upweighting
    2) upweighted based on histogram counts
    3) upweighted analytically via `mass_functions.predict_differential_hmf`
"""

import unittest

import jax.numpy as jnp
import jax.random
import numpy as np

from .... import mc_subhalos
from ....sumstats.diffndhist import tw_ndhist_weighted
from ...smhm_kernels import threeroll_smhm
from ..namedtuple_cat_utils import (
    downsample_and_upweight_cat,
    recursive_namedtuple_cut_and_reindex,
)

Z_OBS = 0.0
VOLUME = 100.0**3
SMF_BINS = jnp.linspace(10, 11.5, 16)
LGMP_MIN = 11.5


def load_mc_halo_cat(seed=0):
    randkey = jax.random.key(seed)

    # Perform initial MC generation slightly below LGMP_MIN
    raw_cat = mc_subhalos(randkey, Z_OBS, LGMP_MIN - 0.2, VOLUME)

    cut = raw_cat.logmp_t_obs[raw_cat.ult_host_indx] >= LGMP_MIN
    return recursive_namedtuple_cut_and_reindex(raw_cat, cut)


def smhm_uparams_tuple_to_array(tup):
    return jnp.array([*tup._asdict().values()])


def smhm_uparams_array_to_tuple(arr):
    return threeroll_smhm.DEFAULT_U_PARAMS._make(arr)


def weighted_triweight_hist(x, bins, weights=None):
    if weights is None:
        weights = jnp.ones_like(x)
    sigma = jnp.full_like(x, bins[1] - bins[0])
    return tw_ndhist_weighted(
        x[:, None], sigma[:, None], weights, bins[:-1, None], bins[1:, None]
    )


def pred_smf(uparams_arr, cat, upweights=None):
    params_tup = threeroll_smhm.get_bounded_params(
        smhm_uparams_array_to_tuple(uparams_arr)
    )
    lgsm = threeroll_smhm.smhm_kernel(params_tup, cat.logmp_t_obs)
    # smf = jnp.histogram(lgsm, bins=SMF_BINS, weights=upweights)[0]
    smf = (
        weighted_triweight_hist(lgsm, SMF_BINS, upweights) / VOLUME / jnp.diff(SMF_BINS)
    )
    return smf


def loss(uparams_arr, cat, truth, upweights=None):
    pred = pred_smf(uparams_arr, cat, upweights=upweights)
    return jnp.sum((pred - truth) ** 2)


def make_jitted_lossfunc_and_gradlossfunc(cat, truth, upweights=None):
    def lossfunc(uparams_arr):
        return loss(uparams_arr, cat, truth, upweights=upweights)

    lossfunc = jax.jit(lossfunc)
    gradlossfunc = jax.jit(jax.grad(lossfunc))
    return lossfunc, gradlossfunc


class TestUpweighting(unittest.TestCase):
    def setUp(self):
        # Set default / shifted shamnet model parameters
        self.params = smhm_uparams_tuple_to_array(threeroll_smhm.DEFAULT_U_PARAMS)
        self.params_hi = self.params + 1.0
        self.params_lo = self.params - 1.0

        # Load catalog
        cat = load_mc_halo_cat()
        self.target_nhost = 1000

        # Downsample catalog via histogram-based upweighting
        cat_hg, upweights_hg = downsample_and_upweight_cat(
            cat, randkey=1, method="histogram", target_nhost=self.target_nhost
        )
        # Downsample catalog via analytic upweighting
        cat_an, upweights_an = downsample_and_upweight_cat(
            cat, randkey=1, method="analytic", target_nhost=self.target_nhost
        )
        self.nhost_hg = (cat_hg.upids == -1).sum()
        self.nhost_an = (cat_an.upids == -1).sum()

        # True SMF for loss computation
        true_smf = pred_smf(self.params, cat)

        # Make jitted loss functions for all three methods
        lossfunc, gradfunc = make_jitted_lossfunc_and_gradlossfunc(cat, true_smf)
        lossfunc_hg, gradfunc_hg = make_jitted_lossfunc_and_gradlossfunc(
            cat_hg, true_smf, upweights=upweights_hg
        )
        lossfunc_an, gradfunc_an = make_jitted_lossfunc_and_gradlossfunc(
            cat_an, true_smf, upweights=upweights_an
        )

        self.jaxfuncs = (
            lossfunc,
            gradfunc,
            lossfunc_hg,
            gradfunc_hg,
            lossfunc_an,
            gradfunc_an,
        )

    def test_downsampled_nhosts(self):
        min_accepted = 0.6 * self.target_nhost
        max_accepted = 1.0 * self.target_nhost
        assert min_accepted < self.nhost_hg < max_accepted
        assert min_accepted < self.nhost_an < max_accepted

    def test_nonzero_loss_and_grad_at_shifted_params(self):
        (lossfunc, gradfunc, lossfunc_hg, gradfunc_hg, lossfunc_an, gradfunc_an) = (
            self.jaxfuncs
        )

        # Full catalog
        assert jnp.any(lossfunc(self.params_hi))
        assert jnp.any(lossfunc(self.params_lo))
        assert jnp.any(gradfunc(self.params_hi))
        assert jnp.any(gradfunc(self.params_lo))

        assert np.all(np.isfinite(gradfunc(self.params_hi)))

        # Histogram-based upsampling
        assert jnp.any(lossfunc_hg(self.params_hi))
        assert jnp.any(lossfunc_hg(self.params_lo))
        assert jnp.any(gradfunc_hg(self.params_hi))
        assert jnp.any(gradfunc_hg(self.params_lo))

        assert np.all(np.isfinite(gradfunc_hg(self.params_hi)))

        # Analytic upsampling
        assert jnp.any(lossfunc_an(self.params_hi))
        assert jnp.any(lossfunc_an(self.params_lo))
        assert jnp.any(gradfunc_an(self.params_hi))
        assert jnp.any(gradfunc_an(self.params_lo))

        assert np.all(np.isfinite(gradfunc_an(self.params_hi)))

    def test_histogram_method_is_all_close(self):
        (lossfunc, gradfunc, lossfunc_hg, gradfunc_hg, _, _) = self.jaxfuncs

        # At default params
        assert jnp.allclose(lossfunc(self.params), lossfunc_hg(self.params), 1e-1, 1e-3)
        assert jnp.allclose(gradfunc(self.params), gradfunc_hg(self.params), 1e-1, 1e-3)

        # At upshifted params
        assert jnp.allclose(
            lossfunc(self.params_hi), lossfunc_hg(self.params_hi), 1e-1, 1e-3
        )
        assert jnp.allclose(
            gradfunc(self.params_hi), gradfunc_hg(self.params_hi), 1e-1, 1e-3
        )

        # At downshifted params
        assert jnp.allclose(
            lossfunc(self.params_lo), lossfunc_hg(self.params_lo), 1e-1, 1e-3
        )
        assert jnp.allclose(
            gradfunc(self.params_lo), gradfunc_hg(self.params_lo), 1e-1, 1e-3
        )

    def test_analytic_method_is_all_close(self):
        (lossfunc, gradfunc, _, _, lossfunc_an, gradfunc_an) = self.jaxfuncs

        # At default params
        assert jnp.allclose(lossfunc(self.params), lossfunc_an(self.params), 1e-1, 1e-3)
        assert jnp.allclose(gradfunc(self.params), gradfunc_an(self.params), 1e-1, 1e-3)

        # At upshifted params
        assert jnp.allclose(
            lossfunc(self.params_hi), lossfunc_an(self.params_hi), 1e-1, 1e-3
        )
        assert jnp.allclose(
            gradfunc(self.params_hi), gradfunc_an(self.params_hi), 1e-1, 1e-3
        )

        # At downshifted params
        assert jnp.allclose(
            lossfunc(self.params_lo), lossfunc_an(self.params_lo), 1e-1, 1e-3
        )
        assert jnp.allclose(
            gradfunc(self.params_lo), gradfunc_an(self.params_lo), 1e-1, 1e-3
        )
