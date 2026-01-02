""""""

import numpy as np
from jax import random as jran

from diffsky.data_loaders.hacc_utils import haccsims

from .. import lightcone_utils as hlu
from .. import load_lc_cf_synthetic as llcs


def test_load_lc_diffsky_patch_data():
    ran_key = jran.key(0)
    lgmp_min, lgmp_max = 12, 12.1
    sim_name = "LastJourney"
    fn_lc_cores_bpat = "lc_cores-{0}.{1}.hdf5"

    patch_decomposition = hlu.read_hacc_lc_patch_decomposition(sim_name)[0]
    n_patches = patch_decomposition.shape[0]
    all_patches = np.arange(n_patches).astype(int)
    sim = haccsims.simulations["LastJourney"]

    n_step, n_patch = 3, 10
    step_key, patch_key = jran.split(ran_key, 2)
    steps = jran.choice(step_key, sim.cosmotools_steps[2:-1], shape=(n_step,))
    steps = steps.astype(int)
    patches = jran.choice(patch_key, all_patches, shape=(n_patch,)).astype(int)

    for step in steps:
        for lc_patch in patches:
            fn_lc_cores = fn_lc_cores_bpat.format(step, lc_patch)
            args = fn_lc_cores, sim_name, ran_key, lgmp_min, lgmp_max
            lc_data, diffsky_data = llcs.load_lc_diffsky_patch_data(*args)

            theta_lo, theta_hi, phi_lo, phi_hi = [
                patch_decomposition[lc_patch, i] for i in range(1, 5)
            ]
            ra_lo, ra_hi, dec_lo, dec_hi = hlu._get_ra_dec_bounds(
                theta_lo, theta_hi, phi_lo, phi_hi
            )

            patch_decomposition[lc_patch, 1:]
            assert np.all(diffsky_data["ra"] >= ra_lo)
            assert np.all(diffsky_data["ra"] <= ra_hi)

            assert np.all(diffsky_data["dec"] >= dec_lo)
            assert np.all(diffsky_data["dec"] <= dec_hi)

            ra_inferred, dec_inferred = hlu.get_ra_dec(
                lc_data["x"], lc_data["y"], lc_data["z"]
            )
            assert np.allclose(ra_inferred, diffsky_data["ra"], rtol=1e-3)
            assert np.allclose(dec_inferred, diffsky_data["dec"], rtol=1e-3)
