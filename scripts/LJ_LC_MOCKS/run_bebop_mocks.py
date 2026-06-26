#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
# PBS -j oe

# account to charge
#PBS -A galsampler

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=1:mpiprocs=4

#PBS -l walltime=2:00:00

# Load software
source ~/.bash_profile
ml gcc openmpi/5.0.5-ohr7u5x
conda activate diffsky_bebop

cd $PBS_O_WORKDIR

YAML_CONFIG=/home/ahearin/work/random/0626/cosmos_260316.yaml
MOCK_SCRIPT_DIR=/home/ahearin/work/repositories/python/diffsky/scripts/LJ_LC_MOCKS

mpirun -n 4 python $MOCK_SCRIPT_DIR/make_lj_mock.py $YAML_CONFIG
mpirun -n 1 python $MOCK_SCRIPT_DIR/add_galaxy_id_to_mock.py $YAML_CONFIG
mpirun -n 1 python $MOCK_SCRIPT_DIR/inspect_lc_mock.py $YAML_CONFIG
