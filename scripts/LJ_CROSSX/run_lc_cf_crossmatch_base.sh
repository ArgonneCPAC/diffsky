#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A cosmo_ai

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=1:mpiprocs=1

#PBS -l walltime=8:00:00

# Load software
source ~/.bash_profile
cd /home/ahearin/work/random/0826/CROSSX
rsync -avz /home/ahearin/work/repositories/python/diffsky/scripts/LJ_CROSSX/lc_cf_crossmatch_script.py ./

python lc_cf_crossmatch_script.py 0.01 3.0 -istart 0 -iend 5 -drn_out /lcrc/project/halotools/random_data/0826 -machine lcrc

python inspect_lightcone_mock.py /lcrc/project/halotools/random_data/0826