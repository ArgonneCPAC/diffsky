#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
# PBS -j oe

# account to charge
#PBS -A cosmo_ai

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=1:mpiprocs=128

#PBS -l walltime=12:00:00

# Load software
source ~/.bash_profile

echo "Working directory: $PBS_O_WORKDIR"
cd $PBS_O_WORKDIR

echo "Job ID: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Running on nodes: $(cat $PBS_NODEFILE)"

# activate your environemnt
conda activate improv311_lensing

# currently just direct python file, will eventually package this but interpolation itself is fairly simple

# need path to galaxy files (with trailing /), and S for synthetics, C for cores (don't do both!), and then the path to the patch_list
# python /lcrc/project/cosmo_ai/prlarsen/codes_lensing/lj_lensing_script.py /lcrc/project/halotools/random_data/0217/hlwas_cosmos_260215_02_17_2026/ C hlwas_cosmos_260215_02_17_2026_patch_list.txt >a2.out 2>b2.out
python /lcrc/project/cosmo_ai/prlarsen/codes_lensing/lj_lensing_script.py /lcrc/project/halotools/random_data/0217/hlwas_cosmos_260215_02_17_2026/ S hlwas_cosmos_260215_02_17_2026_patch_list.txt >a2.out 2>b2.out
