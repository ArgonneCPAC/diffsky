"""Script to make an SFH mock with DiffstarPop"""

import argparse
import os
import subprocess
from time import time
from collections import OrderedDict, namedtuple
from jax import random as jran
from jax import numpy as jnp
from jax import jit as jjit
from functools import partial
from mpi4py import MPI
import numpy as np

from diffdesi.desi_utils.desi_data import Z_KCORRECT
from diffdesi.dsps_utils.ssp_templates import (
    get_ssp_data_photflux_tables_wave_eff
)
from diffdesi.diffstarpop.diffstarpop_get_loss_data import get_loss_data_smhm
from diffdesi.utils.param_utils_scatter import get_all_diffsky_u_params
from diffdesi.utils.namedtuple_utils_scatter import (
    tuple_to_array,
    array_to_tuple_new,
    register_tuple_new,
)
from diffdesi.utils.target_UM_sat_fracs import (
    UM_z0p5_sat_frac,
)
from diffdesi.utils.cosmos_target_data import get_all_cosmos_target_data
from diffdesi.utils.define_bins import (
    define_cosmos_app_mag_bins,
    define_color_bins,
    define_logsm_bins
)
from diffdesi.phot_utils.pred_phot_singlez_scatter import pred_sumstats_singlez
from diffdesi.multidiff import MultiDiffOnePointModel
from diffdesi.merging.merging_model import DEFAULT_MERGE_U_PARAMS
from diffdesi.singlegal.diffdesi_singlegal_scatter import (
    DEFAULT_SCATTER_U_PARAMS,
    DEFAULT_SPSPOP_U_PARAMS,
)

from diffstarpop.kernels.defaults_tpeak import DEFAULT_DIFFSTARPOP_U_PARAMS
from diffstarpop.loss_kernels.smhm_loss_tpeak import (
    mean_smhm_kern_tobs
)

from diffmah.defaults import MAH_K
from diffmah.diffmah_kernels import mah_halopop, _log_mah_kern

from diffsky.utils import smhm_loss_penalty
from diffsky.data_loaders import hacc_core_utils as hcu
from diffsky.data_loaders import load_hacc_cores as lhc

from dsps.metallicity.umzr import DEFAULT_MZR_U_PARAMS

from haccytrees import Simulation as HACCSim

from jax import config
config.update("jax_enable_x64", True)


OUTPAT_CHUNK_RANK = "sfh_mock_subvol_{0}_chunk_{1}_rank_{2}.hdf5"
OUTPAT_CHUNK = "sfh_mock_subvol_{0}_chunk_{1}.hdf5"
OUTPAT_SUBVOL = "sfh_mock_subvol_{0}.hdf5"

DRN_LJ_POBOY = "/Users/aphearin/work/DATA/LastJourney/coretrees"
DRN_LJ_LCRC = "/lcrc/group/cosmodata/simulations/LastJourney/coretrees/forest"

DRN_LJ_DMAH_POBOY = "/Users/aphearin/work/DATA/LastJourney/diffmah_fits"
DRN_LJ_DMAH_LCRC = "/lcrc/project/halotools/LastJourney/diffmah_fits"

BNPAT_CORE_DATA = "m000p.coreforest.{}.hdf5"

NCHUNKS = 20
NUM_SUBVOLS_LJ = 192


# Params----------------------------------------------------------------------
unbound_params_dict = OrderedDict(
    spspop_u_params=DEFAULT_SPSPOP_U_PARAMS,
    merging_u_params=DEFAULT_MERGE_U_PARAMS,
    diffstarpop_u_params=DEFAULT_DIFFSTARPOP_U_PARAMS,
    scatter_u_params=DEFAULT_SCATTER_U_PARAMS,
    mzr_u_params=DEFAULT_MZR_U_PARAMS
)
UnboundParams = namedtuple(
    "UnboundParams", list(unbound_params_dict.keys()))
all_u_params = UnboundParams(*list(unbound_params_dict.values()))
register_tuple_new(UnboundParams)


# Data-------------------------------------------------------------------
sigma = 0.05
dmag = 0.25
k = 20.0
diff_bins = 0.5
app_lo_z0p3 = 19.3
app_mag_range = 2.1

cosmos_data_drn = '/lcrc/project/halotools/COSMOS/'
cosmos_file_z0p3 = 'cosmos20_ugrizYJHK_photz_sm_0p25_0p35.dat'
cosmos_file_z0p5 = 'cosmos20_ugrizYJHK_photz_sm_0p45_0p55.dat'
cosmos_file_z1p1 = 'cosmos20_ugrizYJHK_photz_sm_1p05_1p15.dat'

cosmos_mag_bins, app_lo_mags = define_cosmos_app_mag_bins(
    dmag, app_lo_z0p3, app_mag_range, diff_bins)
(mag_bins_z0p3, mag_bins_z0p5, mag_bins_z1p1) = cosmos_mag_bins
mag_lobins_z0p3 = mag_bins_z0p3[0:-1]
mag_lobins_z0p5 = mag_bins_z0p5[0:-1]
mag_lobins_z1p1 = mag_bins_z1p1[0:-1]
(app_lo_z0p3, app_lo_z0p5, app_lo_z1p1) = app_lo_mags

color_lobins, dcolor, color_bin_centers = define_color_bins()
(
    gr_lobins_z0p3, ri_lobins_z0p3, iz_lobins_z0p3,
    gr_lobins_z0p5, ri_lobins_z0p5, iz_lobins_z0p5,
    gr_lobins_z1p1, ri_lobins_z1p1, iz_lobins_z1p1
) = color_lobins
(
    dgr_z0p3, dri_z0p3, diz_z0p3,
    dgr_z0p5, dri_z0p5, diz_z0p5,
    dgr_z1p1, dri_z1p1, diz_z1p1
) = dcolor

cosmos_data_all = get_all_cosmos_target_data(
    sigma, dmag, k,
    [mag_lobins_z0p3, mag_lobins_z0p5, mag_lobins_z1p1],
    [app_lo_z0p3, app_lo_z0p5, app_lo_z1p1],
    app_mag_range,
    [gr_lobins_z0p3, gr_lobins_z0p5, gr_lobins_z1p1],
    [dgr_z0p3, dgr_z0p5, dgr_z1p1],
    [ri_lobins_z0p3, ri_lobins_z0p5, ri_lobins_z1p1],
    [dri_z0p3, dri_z0p5, dri_z1p1],
    [iz_lobins_z0p3, iz_lobins_z0p5, iz_lobins_z1p1],
    [diz_z0p3, diz_z0p5, diz_z1p1],
    [0.3, 0.5, 1.1],
    [cosmos_file_z0p3, cosmos_file_z0p5, cosmos_file_z1p1]
    )
(
    true_z0p5_r_lum_func, true_z0p5_i_lum_func,
    true_z0p5_gr_pdf_1, true_z0p5_gr_pdf_2, true_z0p5_gr_pdf_3,
    true_z0p5_ri_pdf_1, true_z0p5_ri_pdf_2, true_z0p5_ri_pdf_3,
    true_z0p5_iz_pdf_1, true_z0p5_iz_pdf_2, true_z0p5_iz_pdf_3
) = cosmos_data_all[1]

mag_bins = mag_bins_z0p5

# SMHM Ridge term
indir = "/lcrc/project/halotools/alarcon/results/random_data_250206_v2/"
nhalos = 100
smhm_loss_data, plot_data = get_loss_data_smhm(indir, nhalos)
logsm_target = smhm_loss_data[-1]
logsm_lobins, dlogsm = define_logsm_bins()


# Cosmos data------------------------------------------------
truth = jnp.array((
    *true_z0p5_r_lum_func,
    *true_z0p5_i_lum_func,
    *UM_z0p5_sat_frac,
    *true_z0p5_gr_pdf_1,
    *true_z0p5_gr_pdf_2,
    *true_z0p5_gr_pdf_3,
    *true_z0p5_ri_pdf_1,
    *true_z0p5_ri_pdf_2,
    *true_z0p5_ri_pdf_3,
    *true_z0p5_iz_pdf_1,
    *true_z0p5_iz_pdf_2,
    *true_z0p5_iz_pdf_3,
))
truth = truth + 1e-10


# Functions--------------------------------------------------------------------
@jjit
def _mse(pred, target):
    frac_diff = (pred - target) / target
    sq_diff = frac_diff * frac_diff

    return sq_diff


@partial(jjit, static_argnums=(1,))
def pred_sumstats_singlez_wrapper(unbound_params, index_obs, extras):

    diffstarpop_u_params = unbound_params.diffstarpop_u_params
    spspop_u_params = unbound_params.spspop_u_params
    dustpop_u_params = spspop_u_params.u_dustpop_params
    burstpop_u_params = spspop_u_params.u_burstpop_params
    scatter_u_params = unbound_params.scatter_u_params
    merging_u_params = unbound_params.merging_u_params
    mzr_u_params = unbound_params.mzr_u_params

    nsubvolumes = extras[-1]

    all_diffsky_u_params = get_all_diffsky_u_params(
        varied_sfh_u_params=diffstarpop_u_params.u_sfh_pdf_cens_params,
        varied_satquench_u_params=diffstarpop_u_params.u_satquench_params,
        varied_avpop_u_params=dustpop_u_params.avpop_u_params,
        varied_deltapop_u_params=dustpop_u_params.deltapop_u_params,
        varied_funopop_u_params=dustpop_u_params.funopop_u_params,
        varied_freqburst_u_params=burstpop_u_params.freqburst_u_params,
        varied_fburstpop_u_params=burstpop_u_params.fburstpop_u_params,
        varied_tburstpop_u_params=burstpop_u_params.tburstpop_u_params,
        varied_merging_u_params=merging_u_params,
        varied_scatter_u_params=scatter_u_params,
        varied_mzr_u_params=mzr_u_params
    )

    predictions = pred_sumstats_singlez(all_diffsky_u_params, index_obs, extras[0:-1])

    logsm_pred = mean_smhm_kern_tobs(unbound_params, smhm_loss_data)/nsubvolumes

    predictions = jnp.append(predictions, logsm_pred)

    return predictions


@jjit
def MSE(predictions):

    predictions = predictions + 1e-10

    n_mag_bins_z0p5 = len(mag_lobins_z0p5)
    n_color_bins = len(gr_lobins_z0p5)
    n_logsm_bins = len(logsm_lobins)

    nbins = 0

    pred_logsm_hist_z0p5 = predictions[nbins:nbins+n_logsm_bins]
    pred_sat_logsm_hist_z0p5 = predictions[nbins+n_logsm_bins:nbins+2*n_logsm_bins]
    nbins = nbins + 2*n_logsm_bins
    pred_r_lum_func_z0p5 = predictions[nbins:nbins+n_mag_bins_z0p5]
    pred_i_lum_func_z0p5 = predictions[nbins+n_mag_bins_z0p5:nbins+2*n_mag_bins_z0p5]
    nbins = nbins+2*n_mag_bins_z0p5

    pred_gr_z0p5_1 = predictions[nbins:nbins+n_color_bins]
    pred_gr_z0p5_2 = predictions[nbins+n_color_bins:nbins+2*n_color_bins]
    pred_gr_z0p5_3 = predictions[nbins+2*n_color_bins:nbins+3*n_color_bins]
    pred_ri_z0p5_1 = predictions[nbins+3*n_color_bins:nbins+4*n_color_bins]
    pred_ri_z0p5_2 = predictions[nbins+4*n_color_bins:nbins+5*n_color_bins]
    pred_ri_z0p5_3 = predictions[nbins+5*n_color_bins:nbins+6*n_color_bins]
    pred_iz_z0p5_1 = predictions[nbins+6*n_color_bins:nbins+7*n_color_bins]
    pred_iz_z0p5_2 = predictions[nbins+7*n_color_bins:nbins+8*n_color_bins]
    pred_iz_z0p5_3 = predictions[nbins+8*n_color_bins:nbins+9*n_color_bins]
    pred_N_z0p5_1 = predictions[nbins+9*n_color_bins:nbins+9*n_color_bins+1]
    pred_N_z0p5_2 = predictions[nbins+9*n_color_bins+1:nbins+9*n_color_bins+2]
    pred_N_z0p5_3 = predictions[nbins+9*n_color_bins+2:nbins+9*n_color_bins+3]

    pred_gr_pdf_z0p5_1 = pred_gr_z0p5_1 / pred_N_z0p5_1
    pred_gr_pdf_z0p5_2 = pred_gr_z0p5_2 / pred_N_z0p5_2
    pred_gr_pdf_z0p5_3 = pred_gr_z0p5_3 / pred_N_z0p5_3
    pred_ri_pdf_z0p5_1 = pred_ri_z0p5_1 / pred_N_z0p5_1
    pred_ri_pdf_z0p5_2 = pred_ri_z0p5_2 / pred_N_z0p5_2
    pred_ri_pdf_z0p5_3 = pred_ri_z0p5_3 / pred_N_z0p5_3
    pred_iz_pdf_z0p5_1 = pred_iz_z0p5_1 / pred_N_z0p5_1
    pred_iz_pdf_z0p5_2 = pred_iz_z0p5_2 / pred_N_z0p5_2
    pred_iz_pdf_z0p5_3 = pred_iz_z0p5_3 / pred_N_z0p5_3
    nbins = nbins+9*n_color_bins+3

    pred_sat_frac_z0p5 = pred_sat_logsm_hist_z0p5 / pred_logsm_hist_z0p5

    preds = jnp.array((
        *pred_r_lum_func_z0p5,
        *pred_i_lum_func_z0p5,
        pred_sat_frac_z0p5,
        *pred_gr_pdf_z0p5_1,
        *pred_gr_pdf_z0p5_2,
        *pred_gr_pdf_z0p5_3,
        *pred_ri_pdf_z0p5_1,
        *pred_ri_pdf_z0p5_2,
        *pred_ri_pdf_z0p5_3,
        *pred_iz_pdf_z0p5_1,
        *pred_iz_pdf_z0p5_2,
        *pred_iz_pdf_z0p5_3,
    ))

    logsm_pred = predictions[nbins:]
    ridge = smhm_loss_penalty(logsm_pred, logsm_target, 1.0, dlgsm_max=0.5, h=0.1)

    sq_diff = _mse(preds, truth)
    sq_diff = jnp.append(sq_diff, ridge**2)
    Loss = jnp.mean(sq_diff)

    return Loss


# Instead: MPI-compatible loss calculator
# =======================================
# @jax.tree_util.register_pytree_node_class
class MPILossCalc(MultiDiffOnePointModel):
    # @jjit
    def calc_partial_sumstats_from_params(self, flat_uparams):
        namedtuple_uparams = array_to_tuple_new(flat_uparams, UnboundParams)
        return pred_sumstats_singlez_wrapper(namedtuple_uparams, self.static_data, self.dynamic_data)

    # @jjit
    def calc_loss_from_sumstats(self, sumstats):
        return MSE(sumstats)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()
    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", help="Output directory")
    parser.add_argument('ran_int', help='Random key')
    parser.add_argument('nstep', help='Number of steps')
    parser.add_argument(
        "-redshift", help="redshift of output mock", type=float, default=0.0
    )
    parser.add_argument("-indir_cores", help="Drn of HACC core data", default=None)
    parser.add_argument("-indir_diffmah", help="Drn of diffmah data", default=None)
    parser.add_argument("-sim_name", help="Simulation name", default="LastJourney")
    parser.add_argument(
        "-machine",
        help="Machine name",
        default="poboy",
        type=str,
        choices=["lcrc", "poboy"],
    )
    parser.add_argument(
        "-outbase", help="Basename of the output hdf5 file", default="sfh_mock.hdf5"
    )
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)
    parser.add_argument("-istart", help="First subvolume in loop", type=int, default=0)
    parser.add_argument(
        "-iend", help="Last subvolume in loop", type=int, default=NUM_SUBVOLS_LJ
    )
    parser.add_argument("-nchunks", help="Number of chunks", type=int, default=NCHUNKS)
    parser.add_argument(
        "-num_subvols_tot", help="Total # subvols", type=int, default=NUM_SUBVOLS_LJ
    )

    args = parser.parse_args()
    redshift = args.redshift
    indir_cores = args.indir_cores
    indir_diffmah = args.indir_diffmah
    sim_name = args.sim_name
    machine = args.machine
    istart, iend = args.istart, args.iend
    ran_int = int(args.ran_int)
    nstep = int(args.nstep)

    ran_key = jran.PRNGKey(ran_int)

    num_subvols_tot = args.num_subvols_tot  # needed for string formatting
    outdir = args.outdir
    outbase = args.outbase
    nchunks = args.nchunks

    sim = HACCSim.simulations[sim_name]
    zarr_sim = sim.step2z(np.array(sim.cosmotools_steps))
    iz_obs = np.argmin(np.abs(redshift - zarr_sim))

    nchar_chunks = len(str(nchunks))

    os.makedirs(outdir, exist_ok=True)

    rank_key = jran.key(rank)

    if args.machine == "poboy":
        indir_cores = DRN_LJ_POBOY
        indir_diffmah = DRN_LJ_DMAH_POBOY
    elif args.machine == "lcrc":
        indir_cores = DRN_LJ_LCRC
        indir_diffmah = DRN_LJ_DMAH_LCRC
    else:
        raise ValueError("Unrecognized machine name")

    # Precompute ssp obsmags---------------------------------------------------
    cosmos_ssp_stuff = get_ssp_data_photflux_tables_wave_eff(
        cosmos_data_drn,
        'COSMOS',
        redshift,
        COSMOLOGY,  # replace with LJ cosmology
        Z_KCORRECT
    )
    (
        cosmos_ssp_data,
        cosmos_ssp_obs_photflux_table,
        cosmos_ssp_obs_photflux_nodimming_table,
        cosmos_wave_eff_aa_obsmag,
        cosmos_wave_eff_aa_kcorrect
    ) = cosmos_ssp_stuff  

    start = time()

    if args.test:
        subvolumes = [0]
        chunks = [0, 1]
    else:
        subvolumes = np.arange(istart, iend + 1).astype(int)
        chunks = np.arange(nchunks).astype(int)
    subvolumes = sorted(subvolumes)

    all_avail_subvol_fnames = lhc._get_all_avail_basenames(
        indir_cores, BNPAT_CORE_DATA, subvolumes
    )

    for isubvol in subvolumes:
        isubvol_start = time()

        subvol_str = f"{isubvol}"
        bname_core_data = BNPAT_CORE_DATA.format(subvol_str)
        fn_data = os.path.join(indir_cores, bname_core_data)

        for chunknum in chunks:
            comm.Barrier()
            rank_key, chunk_key_for_rank = jran.split(rank_key, 2)
            ichunk_start = time()

            diffsky_data = lhc.load_diffsky_data_per_rank(
                sim_name,
                isubvol,
                chunknum,
                nchunks,
                iz_obs,
                chunk_key_for_rank,
                indir_cores,
                indir_diffmah,
                comm=MPI.COMM_WORLD,
            )
            fb, lgt0 = hcu.get_diffstar_cosmo_quantities(sim_name)

            MC = 0
            key = jran.PRNGKey(758493)
            nhalos = len(diffsky_data["subcat"].logmp0)
            random_draw = jran.uniform(key, shape=(nhalos,))

            comm.Barrier()

            # Loss data
            lossdata = (
                diffsky_data["subcat"].mah_params,
                diffsky_data["subcat"].logmp0,
                ran_key,
                diffsky_data["tarr"],
                COSMOLOGY,  # replace with LJ cosmology
                lgt0,
                FB,  # replace with LJ baryon fraction
                cosmos_ssp_data,
                cosmos_ssp_obs_photflux_table,
                cosmos_wave_eff_aa_obsmag,
                redshift,
                random_draw,
                diffsky_data["subcat"].logmp_pen_inf,
                diffsky_data["subcat"].logmp_ult_inf,
                diffsky_data["subcat"].logmhost_pen_inf,
                diffsky_data["subcat"].logmhost_ult_inf,
                diffsky_data["tarr"][-1],
                diffsky_data["subcat"].t_pen_inf,
                diffsky_data["subcat"].t_ult_inf,
                diffsky_data["subcat"].upids,
                diffsky_data["subcat"].pen_host_indx,
                diffsky_data["subcat"].ult_host_indx,
                MC,
                sigma,
                color_lobins,
                dcolor,
                comoving_volume,  # replace with comoving volume
                mag_bins_z0p5,
                dmag,
                logsm_lobins,
                dlogsm,
                nsubvolumes,  # replace with number of subvolumes or processes
            )

            chunknum_str = f"{chunknum:0{nchar_chunks}d}"
            bname = OUTPAT_CHUNK_RANK.format(subvol_str, chunknum_str, rank)
            rank_outname = os.path.join(outdir, bname)

            comm.Barrier()
            ichunk_end = time()

            if rank == 0:
                chunk_fnames = []
                for irank in range(nranks):
                    bname = OUTPAT_CHUNK_RANK.format(subvol_str, chunknum_str, irank)
                    fname = os.path.join(outdir, bname)
                    chunk_fnames.append(fname)
                bnout = OUTPAT_CHUNK.format(subvol_str, chunknum_str)
                fnout = os.path.join(outdir, bnout)
                lhc.collate_hdf5_file_collection(chunk_fnames, fnout)
                bnpat = OUTPAT_CHUNK_RANK.format(subvol_str, chunknum_str, "*")
                fnpat = os.path.join(outdir, bnpat)
                command = "rm -rf " + fnpat
                raw_result = subprocess.check_output(command, shell=True)

            comm.Barrier()
