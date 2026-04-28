""""""

import os
import numpy as np
from glob import glob
from .. import load_lc_mock as llcm

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
MATPLOTLIB_MSG = "Must have matplotlib installed to use this function"

try:
    from astropy.cosmology import FlatLambdaCDM

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
ASTROPY_MSG = "Must have astropy installed to use this function"

Z_MIN, Z_MAX = 0.0, 2.0

MBLUE = "#1f77b4"
MGREEN = "#2ca02c"
MORANGE = "#ff7f0e"
MRED = "#d62728"

HOD_HALOPROPS = [
    "redshift_true",
    "central",
    "logsm_obs",
    "logmp_obs",
    "logmp_obs_host",
    "logssfr_obs",
]


def get_plotting_data_mock(
    *,
    patch_list,
    drn_mock,
    drn_synthetic_cores=None,
    z_min=Z_MIN,
    z_max=Z_MAX,
    sim_name="LastJourney",
    keys=HOD_HALOPROPS,
):
    """"""
    pdata = llcm.load_diffsky_lightcone(
        drn_mock, sim_name, z_min, z_max, patch_list, keys=keys
    )
    fn_lc_mock = glob(os.path.join(drn_mock, "lc_cores*.hdf5"))[0]
    metadata = llcm.load_mock_metadata(fn_lc_mock)

    return pdata, metadata


def plot_smf(
    *,
    pdata,
    metadata,
    z_bin,
    dz=0.25,
    drn_out="",
):
    assert HAS_MATPLOTLIB, MATPLOTLIB_MSG
    assert HAS_ASTROPY, ASTROPY_MSG

    drn_out = drn_out or "."
    os.makedirs(drn_out, exist_ok=True)

    dsps_cosmo_mock = metadata["sim_info"].cosmo_params
    astropy_cosmo = FlatLambdaCDM(H0=100 * dsps_cosmo_mock.h, Om0=dsps_cosmo_mock.Om0)
    area_subvol = 31.0  # deg

    logsm_bins = np.linspace(9, 12, 50)
    dlogsm = np.diff(logsm_bins).mean()

    fig, ax = plt.subplots(1, 1)
    __ = ax.loglog()
    xlabel = ax.set_xlabel(r"$M_{\star}\ [M_{\odot}]$")
    ylabel = ax.set_ylabel(r"$\Phi\ {\rm [1/Mpc^{3}/dex]}$")
    ax.set_title(r"${\rm Stellar\ Mass\ Function}$")

    zlo, zhi = z_bin[0] - dz / 2, z_bin[0] + dz / 2
    msk_z = (pdata["redshift_true"] > zlo) & (pdata["redshift_true"] < zhi)
    y, __ = np.histogram(pdata["logsm_obs"][msk_z], bins=logsm_bins)
    vol_4pi = astropy_cosmo.comoving_volume(zhi) - astropy_cosmo.comoving_volume(zlo)
    vol_subvol = (area_subvol / 40_000) * vol_4pi
    __ = ax.plot(
        10 ** logsm_bins[1:],
        y / dlogsm / vol_subvol,
        drawstyle="steps",
        color=MBLUE,
        label=r"${z=0.3}$",
    )

    zlo, zhi = z_bin[1] - dz / 2, z_bin[1] + dz / 2
    msk_z = (pdata["redshift_true"] > zlo) & (pdata["redshift_true"] < zhi)
    y, __ = np.histogram(pdata["logsm_obs"][msk_z], bins=logsm_bins)
    vol_4pi = astropy_cosmo.comoving_volume(zhi) - astropy_cosmo.comoving_volume(zlo)
    vol_subvol = (area_subvol / 40_000) * vol_4pi
    __ = ax.plot(
        10 ** logsm_bins[1:],
        y / dlogsm / vol_subvol,
        drawstyle="steps",
        color=MGREEN,
        label=r"${z=0.9}$",
    )

    zlo, zhi = z_bin[2] - dz / 2, z_bin[2] + dz / 2
    msk_z = (pdata["redshift_true"] > zlo) & (pdata["redshift_true"] < zhi)
    y, __ = np.histogram(pdata["logsm_obs"][msk_z], bins=logsm_bins)
    vol_4pi = astropy_cosmo.comoving_volume(zhi) - astropy_cosmo.comoving_volume(zlo)
    vol_subvol = (area_subvol / 40_000) * vol_4pi
    __ = ax.plot(
        10 ** logsm_bins[1:],
        y / dlogsm / vol_subvol,
        drawstyle="steps",
        color=MORANGE,
        label=r"${z=1.3}$",
    )

    zlo, zhi = z_bin[3] - dz / 2, z_bin[3] + dz / 2
    msk_z = (pdata["redshift_true"] > zlo) & (pdata["redshift_true"] < zhi)
    y, __ = np.histogram(pdata["logsm_obs"][msk_z], bins=logsm_bins)
    vol_4pi = astropy_cosmo.comoving_volume(zhi) - astropy_cosmo.comoving_volume(zlo)
    vol_subvol = (area_subvol / 40_000) * vol_4pi
    ax.plot(
        10 ** logsm_bins[1:],
        y / dlogsm / vol_subvol,
        drawstyle="steps",
        color=MRED,
        label=r"${z=1.9}$",
    )

    ax.legend()
    fig.savefig(
        "smf_analysis_cosmos_260316_04_26_2026.png",
        bbox_extra_artists=[xlabel, ylabel],
        bbox_inches="tight",
        dpi=200,
    )

    plt.close()

    return fig
