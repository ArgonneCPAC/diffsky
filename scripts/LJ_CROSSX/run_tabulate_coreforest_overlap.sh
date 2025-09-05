#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A galsampler

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=1:mpiprocs=30

#PBS -l walltime=2:00:00

# Load software
source ~/.bash_profile
cd $PBS_O_WORKDIR

rsync /home/ahearin/work/repositories/python/diffsky/scripts/LJ_CROSSX/tabulate_coreforest_overlap.py ./
python tabulate_coreforest_overlap.py
