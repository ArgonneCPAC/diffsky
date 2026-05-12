"""Module implements the `load_cosmos20` function"""

import os

import numpy as np
from jax import numpy as jnp

try:
    from astropy.table import Table

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


Z_MIN, Z_MAX = 0.4, 2.5

HSC_MAG_NAMES = ["HSC_g_MAG", "HSC_r_MAG", "HSC_i_MAG", "HSC_z_MAG", "HSC_y_MAG"]
UVISTA_MAG_NAMES = ["UVISTA_Y_MAG", "UVISTA_J_MAG", "UVISTA_H_MAG", "UVISTA_Ks_MAG"]
COSMOS_TARGET_MAGS = [*HSC_MAG_NAMES, *UVISTA_MAG_NAMES]
MAGI_THRESH = 25.5

NANFILL = -999.0

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


def apply_nan_cuts(cosmos, mag_names=COSMOS_TARGET_MAGS):
    """Remove any galaxy with a NaN in any column storing a target magnitude

    Parameters
    ----------
    cosmos : astropy Table

    mag_names : list of strings

    Returns
    -------
    cosmos : astropy Table
        Catalog after applying a NaN cut on `photoz` and any colname in mag_names

    """
    msk_has_nan = np.isnan(cosmos["photoz"])
    for name in mag_names:
        x = np.nan_to_num(
            cosmos[name], copy=True, nan=NANFILL, posinf=NANFILL, neginf=NANFILL
        )
        msk_has_nan = msk_has_nan | (x == NANFILL)

    cosmos = cosmos[~msk_has_nan]
    return cosmos


def get_is_complete_mask(cosmos, z_min=Z_MIN, z_max=Z_MAX, magi_thresh=MAGI_THRESH):
    """Compute mask to define the redshift and i-mag threshold for our target data

    Parameters
    ----------
    cosmos : astropy Table

    z_min, z_max : float
        Galaxies outside this range will be excluded

    magi_thresh : float
        Galaxies fainter than this apparent magnitude will be excluded

    Returns
    -------
    msk_is_complete : array, dtype bool
        Boolean mask defining galaxies that pass the completeness cut

    """
    msk_redshift = (cosmos["photoz"] > z_min) & (cosmos["photoz"] < z_max)
    msk_i_thresh = cosmos["HSC_i_MAG"] < magi_thresh
    msk_is_complete = msk_redshift & msk_i_thresh
    return msk_is_complete


def get_color_outlier_mask(cosmos, mag_names, p_cut=0.5):
    """Compute mask to define extreme outliers in color space

    Parameters
    ----------
    cosmos : astropy Table

    mag_names : list of strings
        Column names defining the colors for which outliers will be excluded

    p_cut : float, optional
        Value in the range [0, 100] defining a percentile cut

    Returns
    -------
    msk_is_not_outlier : array, dtype bool
        Boolean mask defining galaxies that pass the outlier cut

    """
    msk_is_outlier = np.zeros(len(cosmos)).astype(bool)
    for name0, name1 in zip(mag_names[0:], mag_names[1:]):
        c0 = cosmos[name0]
        c1 = cosmos[name1]
        color = c0 - c1
        lo, hi = np.percentile(color, (p_cut, 100.0 - p_cut))
        msk_is_outlier = msk_is_outlier | (color < lo) | (color > hi)

    msk_is_not_outlier = ~msk_is_outlier

    return msk_is_not_outlier
