""""""

import sys

import h5py
import numpy as np
import pytest
from jax import random as jran

from ....data_loaders.hacc_utils import lightcone_utils as hlu
from ....ellipsoidal_shapes import ellipse_proj_kernels as eproj
from .. import inject_ia_adhoc as inj
from .. import intrinsic_alignment as ia
from .test_intrinsic_alignment import (
    _assert_right_handed_orthonormal,
    _kappa_from_mu,
    _ks_bound,
    _ks_distance,
    _uniform_cdf,
    _watson_abs_t_cdf,
)

EIG = "top_host_infall_fof_halo_eig"


def _make_catalog(n, seed=0, b_over_a=None, c_over_a=None, frac_synthetic=0.2):
    """A small catalog with every column the injection script reads.
    Halo shapes have |eigS1| < |eigS2| < |eigS3|; a fraction of the centrals are
    unresolved, with all-zero eigenvectors"""
    rng = np.random.default_rng(seed)
    host = rng.normal(size=(n, 3)) * 400.0
    central = rng.random(n) < 0.5
    pos = np.where(central[:, None], host, host + rng.normal(size=(n, 3)))
    data = {"central": central.astype(int)}
    for i, c in enumerate("xyz"):
        data[f"{c}_host"] = host[:, i]
        data[f"{c}_nfw"] = pos[:, i]
    data["ra_nfw"], data["dec_nfw"] = (np.asarray(v) for v in hlu.get_ra_dec(*pos.T))

    A, B, C = (np.asarray(v) for v in eproj.mc_orthonormal_vectors(jran.key(seed), n))
    synthetic = central & (rng.random(n) < frac_synthetic)
    for i, (vec, length) in enumerate(((C, 0.3), (B, 0.6), (A, 1.0)), start=1):
        vec = np.where(synthetic[:, None], 0.0, vec * length)
        for j, c in enumerate("XYZ"):
            data[f"{EIG}S{i}{c}"] = vec[:, j]

    for comp in ("disk", "bulge"):
        data[f"r50_{comp}_3d"] = rng.uniform(0.5, 3.0, n)
        ba = rng.uniform(0.4, 1.0, n) if b_over_a is None else np.full(n, b_over_a)
        ca = ba * rng.uniform(0.1, 0.6, n) if c_over_a is None else np.full(n, c_over_a)
        data[f"b_over_a_{comp}"], data[f"c_over_a_{comp}"] = ba, ca
        for col in ("alpha", "beta", "psi", "ellipticity", "r50", "mu_inclination"):
            key = f"{col}_{comp}" if col != "r50" else f"r50_{comp}_2d"
            data[key] = np.full(n, -99.0)
    return data, central, synthetic


def _unit(v):
    """Unit vectors; zero vectors (unresolved halos, centrals' radii) stay zero"""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norm > 0, norm, 1.0)


def _halo_major_axis(data):
    return _unit(np.stack([data[f"{EIG}S3{c}"] for c in "XYZ"], axis=-1))


def _radial_unit_vector(data):
    return _unit(np.stack([data[f"{c}_nfw"] - data[f"{c}_host"] for c in "xyz"], axis=-1))


def test_assert_eigS3_is_major_axis():
    """Passes when |eigS3| is the largest (ties included, e.g. all-zero
    unresolved halos), and raises naming how many galaxies violate it"""
    data, _, synthetic = _make_catalog(1_000)
    assert synthetic.sum() > 0
    inj.assert_eigS3_is_major_axis(data)

    for c in "XYZ":  # synthetic-core sentinel: all nine components -1
        for i in (1, 2, 3):
            data[f"{EIG}S{i}{c}"][:5] = -1.0
    inj.assert_eigS3_is_major_axis(data)

    for c in "XYZ":
        data[f"{EIG}S1{c}"][10:13] *= 100.0
    with pytest.raises(AssertionError, match="for 3 of 1000 galaxies"):
        inj.assert_eigS3_is_major_axis(data)


def test_central_and_satellite_alignment_wrap_the_alignment_functions():
    """central_alignment aligns the major axis to the halo axis, and
    satellite_alignment aligns it radially in a box of the given size"""
    data, _, _ = _make_catalog(5_000)
    key = jran.key(3)
    mu = np.linspace(-0.9, 0.9, 5_000)
    eig3 = [data[f"{EIG}S3{c}"] for c in "XYZ"]
    for u, v in zip(inj.central_alignment(*eig3, mu, key), ia.align_to_halo(key, *eig3, mu)):
        assert np.array_equal(u, v)

    hxyz = [data[f"{c}_host"] for c in "xyz"]
    gxyz = [data[f"{c}_nfw"] for c in "xyz"]
    for box in (np.inf, 250.0):
        got = inj.satellite_alignment(*hxyz, *gxyz, 0.5, key, box_size_in_cMpc=box)
        expected = ia.align_radially(key, *hxyz, *gxyz, 0.5, np.full(3, box))
        for u, v in zip(got, expected):
            assert np.array_equal(u, v)


def test_alignment_worker_outputs_are_consistent():
    """Returned shapes respect their bounds and equal an independent projection
    of the returned 3D axes onto the (West, North) plane of each galaxy; the
    inclinations are the cosine of the angle between the line of sight and the
    disk normal (minor axis) or bulge major axis; the 3D axis ratios are the
    catalog's"""
    n = 20_000
    data, central, synthetic = _make_catalog(n)
    disk, bulge, axes = inj.alignment_worker(data, 0.6, 0.6, central, synthetic)
    _assert_right_handed_orthonormal(axes["gal_A"], axes["gal_B"], axes["gal_C"])

    west, north, _ = eproj.observer_frame_axes_from_pos_3D(data["x_nfw"], data["y_nfw"], data["z_nfw"])
    los = np.cross(np.asarray(west), np.asarray(north))
    for comp, e2d, axis in (("disk", disk, axes["gal_C"]), ("bulge", bulge, axes["gal_A"])):
        for p, pname in zip(e2d, e2d._fields):
            assert np.all(np.isfinite(p)), pname
            assert len(p) == n
        assert np.all(e2d.psi >= -np.pi / 2) & np.all(e2d.psi < np.pi / 2)
        assert np.all(e2d.ellipticity >= 0) & np.all(e2d.ellipticity < 1)
        assert np.all(e2d.beta <= e2d.alpha)
        assert np.all(e2d.mu_inclination >= 0) & np.all(e2d.mu_inclination <= 1)
        assert np.allclose(e2d.mu_inclination, np.abs(np.sum(np.asarray(axis) * los, axis=-1)), atol=1e-12)
        assert np.array_equal(e2d.b_over_a, data[f"b_over_a_{comp}"])
        assert np.array_equal(e2d.c_over_a, data[f"c_over_a_{comp}"])

        a = data[f"r50_{comp}_3d"]
        expected = eproj.compute_ellipse2d_in_sim_frame(
            axes["gal_A"], axes["gal_B"], axes["gal_C"],
            a, a * data[f"b_over_a_{comp}"], a * data[f"c_over_a_{comp}"],
            west, north,
        )
        for field in ("alpha", "beta", "ellipticity", "A", "B", "C"):
            assert np.allclose(getattr(e2d, field), getattr(expected, field), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("mu", [0.0, 0.9])
def test_alignment_worker_alignment_statistics(mu):
    """Resolved centrals follow Dimroth-Watson about the halo major axis,
    satellites about the radial direction, and unresolved centrals are random"""
    n = 60_000
    data, central, synthetic = _make_catalog(n, seed=1)
    _, _, axes = inj.alignment_worker(data, mu, mu, central, synthetic)
    major = np.asarray(axes["gal_A"])
    cdf = _watson_abs_t_cdf(_kappa_from_mu(mu))

    resolved = central & ~synthetic
    t_cen = np.abs(np.sum(major * _halo_major_axis(data), axis=-1))[resolved]
    assert _ks_distance(t_cen, cdf) < _ks_bound(t_cen.size)

    t_sat = np.abs(np.sum(major * _radial_unit_vector(data), axis=-1))[~central]
    assert _ks_distance(t_sat, cdf) < _ks_bound(t_sat.size)

    assert _ks_distance(major[synthetic, 2], _uniform_cdf(-1, 1)) < _ks_bound(synthetic.sum())


def test_alignment_worker_position_angles_on_the_sky():
    """Elongated galaxies strongly aligned with their halos have position angles
    close to that of the projected halo axis; without alignment the position
    angles are uniform and unrelated to the halo"""
    n = 40_000
    data, central, synthetic = _make_catalog(n, seed=2, b_over_a=0.3, c_over_a=0.25)
    west, north, _ = (np.asarray(v) for v in eproj.observer_frame_axes_from_ra_dec(data["ra_nfw"], data["dec_nfw"]))
    h = _halo_major_axis(data)
    halo_pa = np.arctan2(np.sum(h * north, axis=-1), np.sum(h * west, axis=-1))
    resolved = central & ~synthetic

    def _offset(psi):
        return np.mod(np.asarray(psi) - halo_pa + np.pi / 2, np.pi) - np.pi / 2

    disk, _, _ = inj.alignment_worker(data, 0.95, 0.95, central, synthetic)
    assert np.median(np.abs(_offset(disk.psi)[resolved])) < np.deg2rad(15)

    disk, _, _ = inj.alignment_worker(data, 0.0, 0.0, central, synthetic)
    uniform = _uniform_cdf(-np.pi / 2, np.pi / 2)
    assert _ks_distance(_offset(disk.psi)[resolved], uniform) < _ks_bound(resolved.sum())
    assert _ks_distance(disk.psi, uniform) < _ks_bound(n)


def _write_catalog(fn, data):
    with h5py.File(fn, "w") as f:
        f.attrs["root_note"] = "kept"
        grp = f.create_group("data")
        for key, val in data.items():
            grp.create_dataset(key, data=val)
            grp[key].attrs["unit"] = "Mpc"
            grp[key].attrs["description"] = f"column {key}"
        meta = f.create_group("metadata")
        meta.attrs["mock_version_name"] = "test_mock"
        meta["z_phot_table"] = np.linspace(0.0, 3.0, 7)
        meta.create_group("cosmology").attrs["Om0"] = 0.31


def _run_main(monkeypatch, fn_in, fn_out, *extra):
    monkeypatch.setattr(sys, "argv", ["inject_ia_adhoc", str(fn_in), str(fn_out), *extra])
    inj.main()


def test_main_writes_a_complete_catalog(tmp_path, monkeypatch):
    """The command-line script rewrites only the shape columns, consistently,
    keeps every other column, attribute and metadata group, and is
    reproducible for a fixed seed"""
    n = 3_000
    data, _, _ = _make_catalog(n)
    fn_in, fn_out = tmp_path / "in.hdf5", tmp_path / "out.hdf5"
    _write_catalog(fn_in, data)
    _run_main(monkeypatch, fn_in, fn_out, "--seed", "7")

    shape_cols = {
        f"{col}_{comp}"
        for col in ("alpha", "beta", "psi", "ellipticity", "mu_inclination")
        for comp in ("disk", "bulge")
    }
    shape_cols |= {"r50_disk_2d", "r50_bulge_2d"}
    with h5py.File(fn_in, "r") as fi, h5py.File(fn_out, "r") as fo:
        assert set(fo["data"].keys()) == set(fi["data"].keys())
        for key in fi["data"]:
            assert dict(fo["data"][key].attrs) == dict(fi["data"][key].attrs)
            if key in shape_cols:
                assert not np.array_equal(fo["data"][key][:], fi["data"][key][:]), key
            else:
                assert np.array_equal(fo["data"][key][:], fi["data"][key][:]), key
        for comp in ("disk", "bulge"):
            alpha = fo["data"][f"alpha_{comp}"][:]
            beta = fo["data"][f"beta_{comp}"][:]
            assert np.allclose(fo["data"][f"r50_{comp}_2d"][:], 0.765 * alpha)
            assert np.allclose(fo["data"][f"ellipticity_{comp}"][:], 1 - beta / alpha)

        assert dict(fo.attrs) == dict(fi.attrs)
        assert dict(fo["metadata"].attrs) == dict(fi["metadata"].attrs)
        assert np.array_equal(fo["metadata/z_phot_table"][:], fi["metadata/z_phot_table"][:])
        assert fo["metadata/cosmology"].attrs["Om0"] == 0.31

    fn_again, fn_other = tmp_path / "again.hdf5", tmp_path / "other.hdf5"
    _run_main(monkeypatch, fn_in, fn_again, "--seed", "7")
    _run_main(monkeypatch, fn_in, fn_other, "--seed", "8")
    with h5py.File(fn_out, "r") as f1, h5py.File(fn_again, "r") as f2, h5py.File(fn_other, "r") as f3:
        assert np.array_equal(f1["data/psi_disk"][:], f2["data/psi_disk"][:])
        assert not np.allclose(f1["data/psi_disk"][:], f3["data/psi_disk"][:])


def test_main_refuses_a_catalog_whose_eigS3_is_not_the_major_axis(tmp_path, monkeypatch):
    data, _, _ = _make_catalog(500)
    for c in "XYZ":
        data[f"{EIG}S1{c}"][:4] *= 100.0
    fn_in = tmp_path / "in.hdf5"
    _write_catalog(fn_in, data)
    with pytest.raises(AssertionError, match="for 4 of 500"):
        _run_main(monkeypatch, fn_in, tmp_path / "out.hdf5")
