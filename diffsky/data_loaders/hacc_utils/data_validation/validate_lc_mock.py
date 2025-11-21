""" """

import os

import h5py
import numpy as np
from diffmah import DEFAULT_MAH_PARAMS
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from dsps.cosmology.flat_wcdm import age_at_z

from .... import phot_utils
from ....experimental import dbk_from_mock
from ....experimental import precompute_ssp_phot as psspp
from ....param_utils import diffsky_param_wrapper as dpw
from .. import lc_mock_production as lcmp
from .. import load_flat_hdf5, load_lc_cf

REQUIRED_METADATA_ATTRS = ("creation_date", "README", "mock_version_name")
REQUIRED_SOFTWARE_VERSION_INFO = (
    "diffmah",
    "diffsky",
    "diffstar",
    "dsps",
    "jax",
    "numpy",
)
BNPAT_LC_MOCK = "data-{0}.{1}.diffsky_gals.hdf5"

HLINE = "----------"


def get_lc_mock_data_report(fn_lc_mock):
    report = dict()
    data = load_flat_hdf5(fn_lc_mock, dataset="data")

    msg = check_all_columns_are_finite(fn_lc_mock, data=data)
    if len(msg) > 0:
        report["finite_colums"] = msg

    msg = check_host_pos_is_near_galaxy_pos(fn_lc_mock, data=data)
    if len(msg) > 0:
        report["nfw_host_distance"] = msg

    msg = check_metadata(fn_lc_mock)
    if len(msg) > 0:
        report["metadata"] = msg

    msg = check_all_data_columns_have_metadata(fn_lc_mock)
    if len(msg) > 0:
        report["column_metadata"] = msg

    msg = check_consistent_disk_bulge_knot_luminosities(fn_lc_mock, data=data)
    if len(msg) > 0:
        report["disk_bulge_knot_inconsistency"] = msg

    return report


def write_lc_mock_report_to_disk(report, fn_lc_mock, drn_report):
    if len(report) > 0:
        bn_report = os.path.basename(fn_lc_mock).replace(".hdf5", ".report.txt")
        fn_report = os.path.join(drn_report, bn_report)
        with open(fn_report, "w") as fn_out:
            for test_key, test_result in report.items():
                fn_out.write(test_key + "\n")
                for line in test_result:
                    fn_out.write(line + "\n")
                fn_out.write(HLINE + "\n")


def check_all_columns_are_finite(fn_lc_mock, data=None):
    bn = os.path.basename(fn_lc_mock)

    if data is None:
        data = load_flat_hdf5(fn_lc_mock, dataset="data")

    msg = []
    for key, arr in data.items():
        if not np.all(np.isfinite(arr)):
            s = f"Column {key} in {bn} has either NaN or inf"
            msg.append(s)

    return msg


def check_all_data_columns_have_metadata(fn_lc_mock):

    msg = []
    with h5py.File(fn_lc_mock, "r") as hdf:
        for key in hdf["data"].keys():
            try:
                unit = hdf["data/" + key].attrs["unit"]
                assert len(unit) > 0
                description = hdf["data/" + key].attrs["description"]
                assert len(description) > 0
            except (KeyError, AssertionError):
                s = f"{key} is missing metadata"
                msg.append(s)
    return msg


def check_metadata(fn_lc_mock):

    msg = []
    with h5py.File(fn_lc_mock, "r") as hdf:
        try:
            # Check all scalar metadata
            avail_medata_attrs = list(hdf["metadata"].attrs.keys())
            assert set(avail_medata_attrs) >= set(REQUIRED_METADATA_ATTRS)

            creation_date = hdf["metadata"].attrs["creation_date"]
            assert len(creation_date) > 0
            mock_version_name = hdf["metadata"].attrs["mock_version_name"]
            assert len(mock_version_name.split("_")) > 1
            README = hdf["metadata"].attrs["README"]
            assert "This file contains diffsky" in README

            # Check cosmology metadata
            Om0 = hdf["metadata/cosmology"].attrs["Om0"]
            assert 0 < Om0 < 1
            all_hacc_cosmo_params = ("Ob0", "Om0", "h", "ns", "sigma8", "w0", "wa")
            assert set(all_hacc_cosmo_params) == set(
                hdf["metadata/cosmology"].attrs.keys()
            )

            # Check nbody_info metadata
            expected_info = ("Lbox", "n_particles", "particle_mass", "sim_name")
            assert set(expected_info) == set(hdf["metadata/nbody_info"].attrs.keys())
            Lbox = hdf["metadata/nbody_info"].attrs["Lbox"]
            assert Lbox > 0
            n_particles = hdf["metadata/nbody_info"].attrs["n_particles"]
            assert n_particles > 0
            particle_mass = hdf["metadata/nbody_info"].attrs["particle_mass"]
            assert particle_mass > 0
            sim_name = hdf["metadata/nbody_info"].attrs["sim_name"]
            assert len(sim_name) > 0

            # Check software_version_info metadata
            avail_software_versions = list(
                hdf["metadata/software_version_info"].attrs.keys()
            )
            assert set(avail_software_versions) == set(REQUIRED_SOFTWARE_VERSION_INFO)

            # Check z_phot_table is reasonable
            z_phot_table = lcmp.load_diffsky_z_phot_table(fn_lc_mock)
            assert z_phot_table.size >= 2
            assert np.all(z_phot_table > -1)
            assert np.all(z_phot_table < 100)

            # Check has ssp_data
            drn_mock = os.path.dirname(fn_lc_mock)
            check_has_ssp_data(drn_mock, mock_version_name)
            check_has_transmission_curves(drn_mock, mock_version_name)

            # Check has param_collection
            check_has_param_collection(drn_mock, mock_version_name)

            # Check has t_table used to tabulate sfh_table
            check_has_t_table(drn_mock, mock_version_name, sim_name)

            # Check photometry in mock agrees with recomputed results
            check_recomputed_photometry(fn_lc_mock)

        except:  # noqa
            s = "metadata is incorrect"
            msg.append(s)

    return msg


def check_host_pos_is_near_galaxy_pos(fn_lc_mock, data=None):
    """host position should be reasonably close to galaxy position"""
    if data is None:
        data = load_flat_hdf5(fn_lc_mock)

    bn = os.path.basename(fn_lc_mock)

    host_x = data["x"][data["top_host_idx"]]
    host_y = data["y"][data["top_host_idx"]]
    host_z = data["z"][data["top_host_idx"]]
    dx = data["x_nfw"] - host_x
    dy = data["y_nfw"] - host_y
    dz = data["z_nfw"] - host_z
    host_dist = np.sqrt(dx**2 + dy**2 + dz**2)

    msg = []
    n_very_far = np.sum(host_dist > 5)
    if n_very_far > 10:
        s = f"{n_very_far} galaxies in {bn} with "
        s += "unexpectedly large xyz distance from top_host_idx"
        msg.append(s)

    msk_cen = data["central"]
    mean_sat_dist = np.abs(np.mean(host_dist[~msk_cen]))
    std_sat_dist = np.std(host_dist[~msk_cen])
    if mean_sat_dist > 1.0:
        s = f"<dist_sat>={mean_sat_dist:.2f} Mpc/h is unexpectedly large"
        msg.append(s)
    if std_sat_dist > 1.0:
        s = f"std(dist_sat)={std_sat_dist:.2f} Mpc/h is unexpectedly large"
        msg.append(s)

    return msg


def check_has_t_table(drn_mock, mock_version_name, sim_name):
    t_table = lcmp.load_diffsky_t_table(drn_mock, mock_version_name)
    sim_info = load_lc_cf.get_diffsky_info_from_hacc_sim(sim_name)

    assert t_table.size == lcmp.N_T_TABLE
    assert np.allclose(t_table[0], lcmp.T_TABLE_MIN, rtol=1e-3)
    assert np.allclose(t_table[-1], 10**sim_info.lgt0, rtol=1e-3)


def check_has_param_collection(drn_mock, mock_version_name):
    param_collection = lcmp.load_diffsky_param_collection(drn_mock, mock_version_name)
    assert len(param_collection) > 0
    param_arr = dpw.unroll_param_collection_into_flat_array(*param_collection)
    pnames = dpw.get_flat_param_names()
    assert len(pnames) == len(param_arr)


def check_has_ssp_data(drn_mock, mock_version_name):
    ssp_data = lcmp.load_diffsky_ssp_data(drn_mock, mock_version_name)
    assert np.all(np.isfinite(ssp_data.ssp_wave))


def check_has_transmission_curves(drn_mock, mock_version_name):
    tcurves = lcmp.load_diffsky_tcurves(drn_mock, mock_version_name)
    assert len(tcurves._fields) > 0
    for tcurve in tcurves:
        assert tcurve.wave.shape == tcurve.transmission.shape
        assert np.all(tcurve.transmission >= 0)
        assert np.all(tcurve.transmission <= 1)


def check_consistent_disk_bulge_knot_luminosities(
    fn_lc_mock, data=None, filter_nickname="lsst_u"
):
    if data is None:
        data = load_flat_hdf5(fn_lc_mock, dataset="data")

    # Enforce that luminosity of disk+bulge+knots equals composite luminosity
    a = 10 ** (-0.4 * data[f"{filter_nickname}_bulge"])
    b = 10 ** (-0.4 * data[f"{filter_nickname}_disk"])
    c = 10 ** (-0.4 * data[f"{filter_nickname}_knots"])
    mtot = -2.5 * np.log10(a + b + c)

    msg = []
    mean_diff_tol = 0.05
    mean_diff = np.mean(mtot - data["lsst_u"])
    if mean_diff > mean_diff_tol:
        s = "disk/bulge/knot luminosities inconsistent with total"
        msg.append(s)
    return msg


def check_recomputed_photometry(fn_lc_mock):
    """Recompute first N_TEST=50 galaxies photometry and enforce agreement"""
    with h5py.File(fn_lc_mock, "r") as hdf:
        mock_version_name = hdf["metadata"].attrs["mock_version_name"]

    N_TEST = 50
    drn_mock = os.path.dirname(fn_lc_mock)
    tcurves = lcmp.load_diffsky_tcurves(drn_mock, mock_version_name)
    t_table = lcmp.load_diffsky_t_table(drn_mock, mock_version_name)
    ssp_data = lcmp.load_diffsky_ssp_data(drn_mock, mock_version_name)
    sim_info = lcmp.load_diffsky_sim_info(fn_lc_mock)

    mock = load_flat_hdf5(fn_lc_mock, dataset="data", iend=N_TEST)
    n_gals = mock["redshift_true"].size

    mah_params = DEFAULT_MAH_PARAMS._make(
        [mock[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make(
        [mock[key] for key in DEFAULT_DIFFSTAR_PARAMS._fields]
    )
    param_collection = lcmp.load_diffsky_param_collection(drn_mock, mock_version_name)
    t_obs = age_at_z(mock["redshift_true"], *sim_info.cosmo_params)

    _msk_q = mock["mc_sfh_type"].reshape((n_gals, 1))
    delta_scatter = np.where(
        _msk_q == 0, mock["delta_scatter_q"], mock["delta_scatter_ms"]
    )

    # Precompute photometry at each element of the redshift table
    z_phot_table = lcmp.load_diffsky_z_phot_table(fn_lc_mock)
    wave_eff_table = phot_utils.get_wave_eff_table(z_phot_table, tcurves)

    precomputed_ssp_mag_table = psspp.get_precompute_ssp_mag_redshift_table(
        tcurves, ssp_data, z_phot_table, sim_info.cosmo_params
    )

    args = (
        mock["redshift_true"],
        t_obs,
        mah_params,
        mock["logmp0"],
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
        sfh_params,
        param_collection.mzr_params,
        param_collection.spspop_params,
        param_collection.scatter_params,
        param_collection.ssperr_params,
        sim_info.cosmo_params,
        sim_info.fb,
        mock["uran_av"],
        mock["uran_delta"],
        mock["uran_funo"],
        delta_scatter,
        mock["mc_sfh_type"],
        mock["fknot"],
    )
    phot_info = dbk_from_mock._disk_bulge_knot_phot_from_mock(*args)

    RTOL = 1e-2
    ATOL = 0.1
    for i, tcurve_name in enumerate(tcurves._fields):
        assert np.allclose(mock[tcurve_name], phot_info["obs_mags"][:, i], rtol=RTOL)
        assert np.allclose(mock[tcurve_name], phot_info["obs_mags"][:, i], atol=ATOL)

        assert np.allclose(
            mock[tcurve_name + "_bulge"],
            phot_info["obs_mags" + "_bulge"][:, i],
            rtol=RTOL,
        )
        assert np.allclose(
            mock[tcurve_name + "_disk"],
            phot_info["obs_mags" + "_disk"][:, i],
            rtol=RTOL,
        )
        assert np.allclose(
            mock[tcurve_name + "_knots"],
            phot_info["obs_mags" + "_knots"][:, i],
            rtol=RTOL,
        )

        assert np.allclose(
            mock[tcurve_name + "_bulge"],
            phot_info["obs_mags" + "_bulge"][:, i],
            atol=ATOL,
        )
        assert np.allclose(
            mock[tcurve_name + "_disk"],
            phot_info["obs_mags" + "_disk"][:, i],
            atol=ATOL,
        )
        assert np.allclose(
            mock[tcurve_name + "_knots"],
            phot_info["obs_mags" + "_knots"][:, i],
            atol=ATOL,
        )
