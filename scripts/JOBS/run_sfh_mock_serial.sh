#!/bin/bash

# join error into standard out file <job_name>.o<job_id>
#PBS -j oe

# account to charge
#PBS -A cosmo_ai

# allocate {select} nodes, each with {mpiprocs} MPI processes
#PBS -l select=1:mpiprocs=1

#PBS -l walltime=12:00:00

# Load software
source ~/.bash_profile
conda activate improv311

cd /home/ahearin/work/random/0626
python make_sfh_lc_mock_lj_serial.py lcrc 0.001 3.0 0 1 /lcrc/project/halotools/random_data/0626_bebop
python make_sfh_lc_mock_lj_serial.py lcrc 0.001 3.0 1 2 /lcrc/project/halotools/random_data/0626_bebop
python make_sfh_lc_mock_lj_serial.py lcrc 0.001 3.0 2 3 /lcrc/project/halotools/random_data/0626_bebop
python make_sfh_lc_mock_lj_serial.py lcrc 0.001 3.0 3 4 /lcrc/project/halotools/random_data/0626_bebop
python make_sfh_lc_mock_lj_serial.py lcrc 0.001 3.0 4 5 /lcrc/project/halotools/random_data/0626_bebop
python make_sfh_lc_mock_lj_serial.py lcrc 0.001 3.0 5 6 /lcrc/project/halotools/random_data/0626_bebop
python make_sfh_lc_mock_lj_serial.py lcrc 0.001 3.0 6 7 /lcrc/project/halotools/random_data/0626_bebop
python make_sfh_lc_mock_lj_serial.py lcrc 0.001 3.0 7 8 /lcrc/project/halotools/random_data/0626_bebop
python make_sfh_lc_mock_lj_serial.py lcrc 0.001 3.0 8 9 /lcrc/project/halotools/random_data/0626_bebop
python make_sfh_lc_mock_lj_serial.py lcrc 0.001 3.0 9 10 /lcrc/project/halotools/random_data/0626_bebop
