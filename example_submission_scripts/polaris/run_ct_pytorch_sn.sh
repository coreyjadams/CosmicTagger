#!/bin/bash -l
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l walltime=3:30:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=home:grand


# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/CosmicTagger
cd ${WORK_DIR}


# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

PRECISION=mixed
FRAMEWORK=torch
LOCAL_BATCH_SIZE=1
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo "Global batch size: ${GLOBAL_BATCH_SIZE}"

# Set up software deps:
module load conda/2022-09-08
conda activate

# Add-ons from conda:
source /home/cadams/Polaris/polaris_conda_2022-09-08-venv/bin/activate

module load cray-hdf5/1.12.1.3

# Env variables for better scaling:
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB

run_id=n_filt64-sn-${PRECISION}_${FRAMEWORK}-lr0.001

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=numa \
python bin/exec.py \
--config-name sn \
run.id=${run_id} \
run.distributed=True \
run.precision=${PRECISION} \
framework=${FRAMEWORK} \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
network.vertex.active=False \
network.classification.active=False \
mode.optimizer.lr_schedule.peak_learning_rate=0.001 \
run.run_length=1 \
run.run_units=epoch
