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
rsync /home/ahearin/work/repositories/python/diffsky/scripts/LJ_LC_MOCKS/make_ou26_mock_batch.py ./

mpirun -n 16 python make_ou26_mock_batch.py lcrc 0 3.0 0 1 /lcrc/project/halotools/random_data/0111 cosmos260105 -cosmos_fit cosmos260105 --no_dbk
mpirun -n 16 python make_ou26_mock_batch.py lcrc 0 3.0 0 1 /lcrc/project/halotools/random_data/0111 cosmos260105 -cosmos_fit cosmos260105 -synthetic_cores 1 -lgmp_min 10.75 -lgmp_max 11.1 --no_dbk

python /home/ahearin/work/repositories/python/diffsky/scripts/LJ_LC_MOCKS/inspect_lc_mock.py /lcrc/project/halotools/random_data/0111/cosmos260105/
python /home/ahearin/work/repositories/python/diffsky/scripts/LJ_LC_MOCKS/inspect_lc_mock.py /lcrc/project/halotools/random_data/0111/synthetic_cores/cosmos260105/
