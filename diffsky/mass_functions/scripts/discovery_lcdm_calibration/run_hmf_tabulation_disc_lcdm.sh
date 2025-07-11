#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A galsampler

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=1:mpiprocs=1

#PBS -l walltime=24:00:00

# Load software
source ~/.bash_profile
conda activate improv311

cd /home/ahearin/work/random/0711
rsync /home/ahearin/work/repositories/python/diffsky/diffsky/mass_functions/scripts/discovery_lcdm_calibration/measure_hmf_target_data_hacc.py ./

python measure_hmf_target_data_hacc.py lcrc
