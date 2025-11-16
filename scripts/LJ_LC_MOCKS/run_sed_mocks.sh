#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
# PBS -j oe

# account to charge
#PBS -A galsampler

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=16:mpiprocs=1

#PBS -l walltime=1:00:00

# Load software
source ~/.bash_profile
conda activate improv311

cd $PBS_O_WORKDIR
rsync /home/ahearin/work/repositories/python/diffsky/scripts/LJ_LC_MOCKS/make_dbk_sed_lc_mock_lj.py ./

mpirun -n 16 python make_dbk_sed_lc_mock_lj.py lcrc 0 3.0 0 2 /lcrc/project/halotools/random_data/1116 smdpl_dr1 -fn_u_params sfh_model -sfh_model smdpl_dr1
mpirun -n 16 python make_dbk_sed_lc_mock_lj.py lcrc 0 3.0 0 2 /lcrc/project/halotools/random_data/1116 smdpl_dr1 -fn_u_params sfh_model -sfh_model smdpl_dr1 -synthetic_cores 1 -lgmp_min 10.75 -lgmp_max 11.1

python /home/ahearin/work/repositories/python/diffsky/scripts/LJ_LC_MOCKS/inspect_lc_mock.py /lcrc/project/halotools/random_data/1116/smdpl_dr1/
python /home/ahearin/work/repositories/python/diffsky/scripts/LJ_LC_MOCKS/inspect_lc_mock.py /lcrc/project/halotools/random_data/1116/synthetic_cores/smdpl_dr1/
