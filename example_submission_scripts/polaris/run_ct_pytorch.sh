#!/bin/sh
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l walltime=3:00:00
#PBS -q workq
#PBS -A datascience

# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/CosmicTagger
cd ${WORK_DIR}

DATA_DIR=/lus/grand/projects/datascience/cadams/datasets/SBND/

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NDEPTH=8

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=2
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo $GLOBAL_BATCH_SIZE

# Set up software deps:
module load conda/2022-07-19
conda activate

# Add-ons from conda:
source /home/cadams/Polaris/polaris_conda_2022-07-19-venv/bin/activate

module load cray-hdf5/1.12.1.3

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind=depth \
python bin/exec.py \
--config-name sn-at \
run.id=polaris_sn-at_${GLOBAL_BATCH_SIZE} \
run.distributed=True \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
data.data_directory=${DATA_DIR} \
run.iterations=4000
