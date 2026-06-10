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

# need path to galaxy files (with trailing /), and S for synthetics, C for cores (don't do both!), and then the path to the patch_list
python lj_lensing_script.py /lcrc/project/halotools/random_data/0605/cosmos_260316_06_05_2026/ C cosmos_260316_patch_list.txt >a2.out 2>b2.out
python lj_lensing_script.py /lcrc/project/halotools/random_data/0605/cosmos_260316_06_05_2026/ S cosmos_260316_patch_list.txt >a2.out 2>b2.out
