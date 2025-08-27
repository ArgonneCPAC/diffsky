""""""

import os
import subprocess
from copy import deepcopy
from glob import glob

import h5py
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from dsps.cosmology import flat_wcdm
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from .. import load_flat_hdf5
from . import hacc_core_utils as hcu
from . import load_hacc_cores as lhc
from .defaults import DIFFMAH_MASS_COLNAME

DEG_PER_RAD = 180 / np.pi
SQDEG_PER_STER = DEG_PER_RAD**2
SQDEG_OF_SPHERE = SQDEG_PER_STER * 4 * np.pi

TOP_HOST_MAH_KEYS = ["top_host_" + key for key in DEFAULT_MAH_PARAMS._fields]
SECONDARY_HOST_MAH_KEYS = ["sec_host_" + key for key in DEFAULT_MAH_PARAMS._fields]

shapes_1 = [f"infall_fof_halo_eigS1{x}" for x in ("X", "Y", "Z")]
shapes_2 = [f"infall_fof_halo_eigS2{x}" for x in ("X", "Y", "Z")]
shapes_3 = [f"infall_fof_halo_eigS3{x}" for x in ("X", "Y", "Z")]
SHAPE_KEYS = [*shapes_1, *shapes_2, *shapes_3]
TOP_HOST_SHAPE_KEYS = ["top_host_" + key for key in SHAPE_KEYS]


LC_PATCH_OUT_KEYS = (
    *DEFAULT_MAH_PARAMS._fields,
    *TOP_HOST_MAH_KEYS,
    *SECONDARY_HOST_MAH_KEYS,
    *TOP_HOST_SHAPE_KEYS,
    "loss",
    "n_points_per_fit",
    "indx_t_ult_inf",
    "indx_t_pen_inf",
    "n_cf_match",
    "mp_obs",
    "mp0",
    "vx",
    "vy",
    "vz",
)
LC_PATCH_OUT_INT_KEYS = (
    "n_points_per_fit",
    "indx_t_ult_inf",
    "indx_t_pen_inf",
    "n_cf_match",
)
LC_SUBVOL_DRNPAT = "subvols_*"
LC_PATCH_DIFFSKY_BNPAT = "lc_cores-{0}.{1}.diffsky_data.hdf5"
LC_PATCH_BNPAT = "lc_cores-{0}.{1}.hdf5"

BNPAT_LC_CFG = "lc_patch_list_{0}_to_{1}.cfg"

try:
    from haccytrees import Simulation as HACCSim

    HAS_HACCYTREES = True
except ImportError:
    HAS_HACCYTREES = False


@jjit
def _jnp_take_kern(arr, indx):
    return arr[indx]


_jnp_take_vmap = jjit(vmap(_jnp_take_kern, in_axes=(0, 0)))


@jjit
def jnp_take_matrix(matrix, indxarr):
    """Retrieve entries indxarr from the second column of the input matrix"""
    return _jnp_take_vmap(matrix, indxarr)


@jjit
def get_theta_phi(x, y, z):
    """Compute lightcone angles from simulation xyz coords

    Parameters
    ----------
    x,y,z : array, shape (n, )
       Cartesian coords in any units

    Returns
    -------
    theta : array, shape (n, )
        colatitude in radians
        0 < θ < π

    phi : array, shape (n, )
        longitude in radians
        0 < φ < 2π

    """
    rsq = x * x + y * y + z * z
    r = jnp.sqrt(rsq)
    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x) + jnp.pi
    return theta, phi


@jjit
def get_ra_dec(x, y, z):
    """Compute ra, dec in degrees from simulation xyz coords

    Parameters
    ----------
    x,y,z : array, shape (n, )
       Cartesian coords in any units

    Returns
    -------
    ra : array, shape (n, )
        0 < ra < 360

    dec : array, shape (n, )
        -90 < dec < 90

    """
    theta, phi = get_theta_phi(x, y, z)
    ra, dec = _get_lon_lat_from_theta_phi(theta, phi)
    return ra, dec


@jjit
def get_ra_dec_from_theta_phi(theta, phi):
    """Change sky coordinates from {theta, phi} (radians) to {ra, dec} (degrees)"""
    return _get_lon_lat_from_theta_phi(theta, phi)


@jjit
def _get_lon_lat_from_theta_phi(theta, phi):
    lon = jnp.degrees(phi)
    lat = 90.0 - jnp.degrees(theta)
    return lon, lat


@jjit
def get_redshift_from_xyz(x_mpch, y_mpch, z_mpch, cosmo_params):
    """Compute redshift from simulation xyz coords

    Parameters
    ----------
    x,y,z : array, shape (n, )
       Cartesian coords in Mpc/h

    Returns
    -------
    redshift : array, shape (n, )

    """
    d_mpch = jnp.sqrt(x_mpch**2 + y_mpch**2 + z_mpch**2)
    d_mpc = d_mpch / cosmo_params.h

    z_table = jnp.linspace(0, 20, 1_000)
    d_table = flat_wcdm.comoving_distance(
        z_table, cosmo_params.Om0, cosmo_params.w0, cosmo_params.wa, cosmo_params.h
    )
    redshift = jnp.interp(d_mpc, d_table, z_table)

    return redshift


def read_lc_ra_dec_patch_decomposition(fn):
    """Read and parse 'lc_cores-decomposition.txt'

    Parameters
    ----------
    fname : string

    Returns
    -------
    patch_decomposition : array shape (n_patches, 5)
        Each row stores (patchnum, theta_low, theta_high, phi_low, phi_high)

        The relationship between {theta, phi} and {ra, dec} is given by the
        lightcone_utils.get_ra_dec_from_theta_phi function

    sky_frac : array shape (n_patches, )
        Size of the sky patch as a fraction of 4π steradians

    solid_angles : array shape (n_patches, )
        Size of the sky patch in deg^2

    """
    collector = []
    with open(fn, "r") as f:
        next(f)
        next(f)
        for raw_line in f:
            line = raw_line.strip().split()
            patch_idx = int(line[0])
            theta_low = float(line[1][1:-1])
            theta_high = float(line[2][:-1])
            phi_low = float(line[3][1:-1])
            phi_high = float(line[4][:-1])
            data = patch_idx, theta_low, theta_high, phi_low, phi_high
            collector.append(data)
    patch_decomposition = np.array(collector)

    sky_frac = []
    solid_angles = []
    for ipatch in range(patch_decomposition.shape[0]):
        dec_min, dec_max, ra_min, ra_max = patch_decomposition[ipatch, 1:]
        solid_angle, fraction = calculate_solid_angle(ra_min, ra_max, dec_min, dec_max)
        sky_frac.append(fraction)
        solid_angles.append(solid_angle)
    sky_frac = np.array(sky_frac)
    solid_angles = np.array(solid_angles)

    return patch_decomposition, sky_frac, solid_angles


def calculate_solid_angle(ra_min, ra_max, dec_min, dec_max, epsilon=0.001):
    """
    Calculate the solid angle of a rectangular patch of sky.

    Parameters:
    -----------
    ra_min, ra_max : floats
        Right ascension range in radians (0 to 2π)

    dec_min, dec_max : floats
        Declination range in radians (0 to π, where 0 is north pole,
        π/2 is equator, and π is south pole)

    Returns:
    --------
    solid_angle : float
        Solid angle in square degrees

    fsky : float
        Fraction of the full sky covered

    """
    # Input validation
    ra_lo, ra_hi = -epsilon, 2 * np.pi + epsilon
    dec_lo, dec_hi = -epsilon, np.pi + epsilon

    if not (ra_lo <= ra_min <= ra_hi and ra_lo <= ra_max <= ra_hi):
        raise ValueError(
            f"RA must be between 0 and 2π radians. Got ra_min={ra_min}, ra_max={ra_max}"
        )
    if not (dec_lo <= dec_min <= dec_hi and dec_lo <= dec_max <= dec_hi):
        raise ValueError(
            f"DEC must be between 0 and π radians. Got dec_min={dec_min}, dec_max={dec_max}"
        )
    if ra_min > ra_max:
        raise ValueError(
            f"ra_min must be less than ra_max. Got ra_min={ra_min}, ra_max={ra_max}"
        )
    if dec_min > dec_max:
        raise ValueError(
            f"dec_min must be less than dec_max. Got dec_min={dec_min}, dec_max={dec_max}"
        )

    ra_min = np.clip(ra_min, 0, 2 * np.pi)
    ra_max = np.clip(ra_max, 0, 2 * np.pi)
    dec_min = np.clip(dec_min, 0, np.pi)
    dec_max = np.clip(dec_max, 0, np.pi)

    delta_ra = ra_max - ra_min
    solid_angle = delta_ra * (np.cos(dec_min) - np.cos(dec_max))  # in steradians
    solid_angle = solid_angle * SQDEG_PER_STER  # in degrees

    fsky = solid_angle / SQDEG_OF_SPHERE

    return solid_angle, fsky


def get_infall_times_lc_shell(forest_matrices, timestep_idx):
    args = (
        forest_matrices["host_row"],
        forest_matrices["central"],
        forest_matrices["top_host_row"],
        forest_matrices["secondary_top_host_row"],
        timestep_idx,
    )
    indx_t_ult_inf, indx_t_pen_inf = lhc.get_infall_time_indices(*args)
    return indx_t_ult_inf, indx_t_pen_inf


def get_diffsky_quantities_for_lc_patch(
    lc_patch_data,
    lc_patch_data_out,
    cf_matrices,
    cf_diffmah_data,
    cf_file_idx,
    cf_indx_t_ult_inf,
    cf_indx_t_pen_inf,
    timestep_idx,
):
    cf_first_row = cf_matrices["absolute_row_idx"][0]
    cf_last_row = cf_matrices["absolute_row_idx"][-1]
    msk_olap = lc_patch_data["coreforest_file_idx"] == cf_file_idx
    msk_olap &= lc_patch_data["coreforest_row_idx"] >= cf_first_row
    msk_olap &= lc_patch_data["coreforest_row_idx"] <= cf_last_row
    mpeak_history = np.maximum.accumulate(cf_matrices[DIFFMAH_MASS_COLNAME], axis=1)
    mp_obs = mpeak_history[:, timestep_idx]
    mp0 = mpeak_history[:, -1]

    n_olap = msk_olap.sum()
    if n_olap > 0:
        olap_chunk_idx = lc_patch_data["coreforest_row_idx"][msk_olap] - cf_first_row

        # Enforce exact match in core_tag
        core_tags_cf = cf_matrices["core_tag"][:, -1][olap_chunk_idx]
        core_tags_lc = lc_patch_data["core_tag"][msk_olap]
        msg = "Mismatched core_tag between coreforest and lc-cores"
        assert np.allclose(core_tags_cf, core_tags_lc), msg

        olap_chunk_ult_host_idx = cf_matrices["top_host_row"][:, timestep_idx][
            olap_chunk_idx
        ]
        olap_chunk_pen_host_idx = cf_matrices["secondary_top_host_row"][
            :, timestep_idx
        ][olap_chunk_idx]

        lc_patch_data_out["mp_obs"][msk_olap] = mp_obs[olap_chunk_idx]
        lc_patch_data_out["mp0"][msk_olap] = mp0[olap_chunk_idx]

        olap_indx_t_ult_inf = cf_indx_t_ult_inf[olap_chunk_idx]
        lc_patch_data_out["indx_t_ult_inf"][msk_olap] = olap_indx_t_ult_inf

        olap_indx_t_pen_inf = cf_indx_t_pen_inf[olap_chunk_idx]
        lc_patch_data_out["indx_t_pen_inf"][msk_olap] = olap_indx_t_pen_inf

        lc_patch_data_out["n_cf_match"][msk_olap] += 1

        # Matchup core velocities
        _keys = ("vx", "vy", "vz")
        _olap_vel = [cf_matrices[key][olap_chunk_idx][:, timestep_idx] for key in _keys]
        for key, v in zip(_keys, _olap_vel):
            lc_patch_data_out[key][msk_olap] = v

        # Matchup mah_params
        _keys = (*DEFAULT_MAH_PARAMS._fields, "loss", "n_points_per_fit")
        _olap_data = [cf_diffmah_data[key][olap_chunk_idx] for key in _keys]
        for key, x in zip(_keys, _olap_data):
            lc_patch_data_out[key][msk_olap] = x

        # Matchup uber host_mah_params
        mah_keys = DEFAULT_MAH_PARAMS._fields
        _olap_ult_host_mah_data = [
            cf_diffmah_data[key][olap_chunk_ult_host_idx] for key in mah_keys
        ]
        for host_key, x in zip(TOP_HOST_MAH_KEYS, _olap_ult_host_mah_data):
            lc_patch_data_out[host_key][msk_olap] = x

        # Matchup uber host shapes
        _olap_ult_host_shape_data = [
            cf_matrices[key][:, timestep_idx][olap_chunk_ult_host_idx]
            for key in SHAPE_KEYS
        ]
        for host_key, x in zip(TOP_HOST_SHAPE_KEYS, _olap_ult_host_shape_data):
            lc_patch_data_out[host_key][msk_olap] = x

        # Matchup secondary host_mah_params
        mah_keys = DEFAULT_MAH_PARAMS._fields
        second_host_mah_keys = SECONDARY_HOST_MAH_KEYS
        _olap_pen_host_data = [
            cf_diffmah_data[key][olap_chunk_pen_host_idx] for key in mah_keys
        ]
        for host_key, x in zip(second_host_mah_keys, _olap_pen_host_data):
            lc_patch_data_out[host_key][msk_olap] = x

    return lc_patch_data_out


def _get_lc_patch_data_out_bname_for_rank(bn_patch, rank):
    bn_out = bn_patch.replace(".hdf5", f".diffsky_data_rank_{rank}.hdf5")
    return bn_out


def load_lc_patch_data_out(drn, bn_patch, rank):
    bn_out = _get_lc_patch_data_out_bname_for_rank(bn_patch, rank)
    fn_out = os.path.join(drn, bn_out)
    lc_patch_data_out = load_flat_hdf5(fn_out)
    return lc_patch_data_out


def initialize_lc_patch_data_out(n_patch):
    lc_patch_data_out = dict()
    for key in LC_PATCH_OUT_KEYS:
        if key in LC_PATCH_OUT_INT_KEYS:
            lc_patch_data_out[key] = np.zeros(n_patch).astype(int)
        else:
            lc_patch_data_out[key] = np.zeros(n_patch).astype(float)

    return lc_patch_data_out


def overwrite_lc_patch_data_out(lc_patch_data_out, drn_out, bn_patch, rank):
    bn_out = _get_lc_patch_data_out_bname_for_rank(bn_patch, rank)
    fn_out = os.path.join(drn_out, bn_out)

    with h5py.File(fn_out, "w") as hdf_out:
        for key in lc_patch_data_out.keys():
            hdf_out[key] = lc_patch_data_out[key]


def get_stepnum_and_skypatch_from_lc_bname(lc_bname):
    """Assumes basename format such as lc_cores-300.0.hdf5"""
    seq = lc_bname.split("-")[1].split(".")
    stepnum = int(seq[0])
    patchnum = int(seq[1])
    return stepnum, patchnum


def get_lc_patches_in_zrange(sim_name, lc_xdict, z_min, z_max, patch_list=None):
    _res = hcu.get_timestep_range_from_z_range(sim_name, z_min, z_max)
    timestep_min, timestep_max = _res[2:]

    lc_patches = []
    for patch_info in lc_xdict.keys():
        stepnum, patchnum = patch_info

        if (stepnum >= timestep_min) & (stepnum <= timestep_max):

            if patch_list is None:
                lc_patches.append(patch_info)
            else:
                if patchnum in patch_list:
                    lc_patches.append(patch_info)

    return lc_patches


def check_lc_cores_diffsky_data(fn):
    report = dict()
    data = load_flat_hdf5(fn)
    n_cf_match_list = np.unique(data["n_cf_match"])
    report["n_cf_match_list"] = n_cf_match_list
    return report


def write_lc_cores_diffsky_data_report_to_disk(report, fnout):
    lc_cf_perfect_match = set(report["n_cf_match_list"]) == {1}
    if not lc_cf_perfect_match:
        fnout = fnout.replace(".hdf5", ".report.txt")
        with open(fnout, "w") as fout:
            fout.write("n_cf_match is not 1 for all objects\n")

    all_good = deepcopy(lc_cf_perfect_match)

    return all_good


def collate_rank_data(drn_in, drn_out, lc_patches, nranks, cleanup=True):
    for patch_info in lc_patches:
        stepnum, patchnum = patch_info
        bn_patch_out = LC_PATCH_DIFFSKY_BNPAT.format(stepnum, patchnum)

        # Collect patch data from all ranks
        data_collector = []
        fname_collector = []
        for rank in range(nranks):
            bname_in = _get_lc_patch_data_out_bname_for_rank(
                LC_PATCH_BNPAT.format(stepnum, patchnum), rank
            )
            fname_in = os.path.join(drn_in, bname_in)
            data_collector.append(load_flat_hdf5(fname_in))
            fname_collector.append(fname_in)

        # Initialize output data
        example_key = LC_PATCH_OUT_KEYS[0]
        n_patch = data_collector[0][example_key].size
        data_out = initialize_lc_patch_data_out(n_patch)

        # Write collated data to disk in a single file
        fn_patch_out = os.path.join(drn_out, bn_patch_out)
        with h5py.File(fn_patch_out, "w") as hdf_out:
            for key in LC_PATCH_OUT_KEYS:
                for rank_data in data_collector:
                    data_out[key] = data_out[key] + rank_data[key]
                hdf_out[key] = data_out[key]

        # Delete temporary files created by each rank
        if cleanup:
            for fn in fname_collector:
                command = f"rm {fn}"
                subprocess.check_output(command, shell=True)


def _check_serial_vs_parallel(drn1, drn2):
    fn_list1 = glob(os.path.join(drn1, LC_PATCH_BNPAT.format("*", "*")))
    discrepant_file_list = []
    for fn1 in fn_list1:
        bn1 = os.path.basename(fn1)
        fn2 = os.path.join(drn2, bn1)
        data1 = load_flat_hdf5(fn1)
        data2 = load_flat_hdf5(fn2)

        for key in data1.keys():
            try:
                assert np.allclose(data1[key], data2[key])
            except AssertionError:
                discrepant_file_list.append(bn1)
    if len(discrepant_file_list) == 0:
        print("Every pair of matching files is identical")
    else:
        print(f"The following files have discrepancies:{discrepant_file_list}")


def make_cfg_file(i, j, drn=""):
    fn = os.path.join(drn, BNPAT_LC_CFG.format(i, j))
    with open(fn, "w") as fout:
        for patch_num in range(i, j + 1):
            fout.write(f"{patch_num}\n")
    return fn


def get_a_range_of_lc_cores_file(bname_lc_cores, sim_name):
    """Get range of scale factor spanned by data in bname_lc_cores"""
    sim = HACCSim.simulations[sim_name]
    steps = np.array(sim.cosmotools_steps)
    aarr = sim.step2a(steps)

    stepnum, lc_patch = [int(s) for s in bname_lc_cores.split("-")[1].split(".")[:-1]]
    indx_step = np.searchsorted(steps, stepnum)
    a_min_expected, a_max_expected = aarr[indx_step], aarr[indx_step + 1]

    return a_min_expected, a_max_expected
    return a_min_expected, a_max_expected
