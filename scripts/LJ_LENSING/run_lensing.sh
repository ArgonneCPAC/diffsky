#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
# PBS -j oe

# account to charge
#PBS -A halotools

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=1:mpiprocs=1

#PBS -l walltime=6:00:00

# Load software and activate conda env
source ~/.bash_profile
ml gcc/13.2.0 openmpi/5.0.6-gcc-13.2.0
conda activate diffsky_improv

echo "Working directory: $PBS_O_WORKDIR"
cd $PBS_O_WORKDIR

echo "Job ID: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Running on nodes: $(cat $PBS_NODEFILE)"

LJ_MOCK_SCRIPT_DIR=/home/ahearin/work/repositories/python/diffsky/scripts/LJ_LC_MOCKS
LJ_LENSING_SCRIPT_DIR=/home/ahearin/work/repositories/python/diffsky/scripts/LJ_LENSING
MOCK_DATA_DIR=/lcrc/project/galsampler/Roman-GRS-PIT/c260710_08_02_2026


# need path to galaxy files (with trailing /), and S for synthetics, C for cores (don't do both!), and then the path to the patch_list
mpirun -n 1 python $LJ_LENSING_SCRIPT_DIR/lj_lensing_script.py $MOCK_DATA_DIR C $LJ_MOCK_SCRIPT_DIR/emlines_info_grs_pit.dat >a2.out 2>b2.out
mpirun -n 1 python $LJ_LENSING_SCRIPT_DIR/lj_lensing_script.py $MOCK_DATA_DIR S $LJ_MOCK_SCRIPT_DIR/emlines_info_grs_pit.dat >a2.out 2>b2.out
