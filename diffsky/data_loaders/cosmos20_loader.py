"""Module implements the `load_cosmos20` function"""

import os

import numpy as np
from jax import numpy as jnp

try:
    from astropy.table import Table

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


COSMOS20_BASENAME = "COSMOS2020_Farmer_processed_hlin.fits"

SKY_AREA = 1.21  # square degrees

NANFILL = -999.0


__all__ = ("load_cosmos20",)


def load_cosmos20(
    drn=None, bn=COSMOS20_BASENAME, apply_cuts=True, mag_lo=-100, mag_hi=5
):
    """Load the COSMOS-20 dataset from disk and calculate quality cuts

    Parameters
    ----------
    drn : string, optional
        Absolute path to directory containing .fits file storing COSMOS-20 dataset
        Default value is os.environ['COSMOS20_DRN'].

        For bash users, add the following line to your `.bash_profile` in order to
        configure the package to use your default dataset location:

        export COSMOS20_DRN="/drn/storing/COSMOS20"

    bn : string, optional
        Absolute path to directory containing .fits file storing COSMOS-20 dataset
        Default value is COSMOS20_BASENAME set at top of module

    apply_cuts : bool, optional
        If True, returned Table will have quality cuts imposed on the data
        Default is True

    mag_lo : int, optional
        Smallest absolute magnitude in any band before galaxy is considered unphysical

    mag_hi : int, optional
        Largest absolute magnitude in any band before galaxy is considered unphysical

    Returns
    -------
    cat : astropy.table.Table
        Table of length ngals

    Notes
    -----
    Quality cuts include lp_type=0 for the `galaxies` flag.
    And for every Mag in the Le Phare absolute magnitudes,
    we require mag_lo < Mag < mag_hi

    """
    if not HAS_ASTROPY:
        raise ImportError("Must have astropy installed to use cosmos20_loader.py")

    if drn is None:
        try:
            drn = os.environ["COSMOS20_DRN"]
        except KeyError:
            msg = "Must set environment variable COSMOS20_DRN or pass drn argument"
            raise KeyError(msg)

    fn = os.path.join(drn, bn)
    cat = Table.read(fn, format="fits", hdu=1)

    if apply_cuts:
        cat_out = Table()
        cuts = []
        sel_galaxies = np.array(cat["lp_type"] == 0).astype(bool)
        cuts.append(sel_galaxies)

        lp_keys = [key for key in cat.keys() if "lp_M" in key]
        for key in lp_keys:
            x = np.nan_to_num(
                cat[key], copy=True, nan=NANFILL, posinf=NANFILL, neginf=NANFILL
            )
            key_finite_msk = np.isfinite(x == NANFILL)
            cuts.append(key_finite_msk)
            cuts.append(x > mag_lo)
            cuts.append(x < mag_hi)

        msk = np.prod(cuts, axis=0).astype(bool)
        for key in cat.keys():
            cat_out[key] = jnp.array(cat[key][msk])

        return cat_out

    else:
        return cat

    return cat
