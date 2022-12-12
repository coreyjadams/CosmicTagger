#!/bin/sh
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A datascience

# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/CosmicTagger
cd ${WORK_DIR}


# Set up software deps:
module load conda/2022-07-19
conda activate

# Add-ons from conda:
source /home/cadams/Polaris/polaris_conda_2022-07-19-venv/bin/activate

module load cray-hdf5/1.12.1.3

python benchmark_performance.py
