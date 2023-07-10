#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:grand

# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/CosmicTagger
cd ${WORK_DIR}

DATA_DIR=/lus/gila/projects/Aurora_deployment/cadams/cosmic_tagger/

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=8
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo $GLOBAL_BATCH_SIZE

# export NVIDIA_TF32_OVERRIDE=0
run_id=single-SPR-fp32-CPU

# CPU params:
export OMP_NUM_THREADS=32
export KMP_BLOCKTIME=0
COMPUTE_MODE=CPU
source /home/cadams/miniconda3/bin/activate
conda activate tf-cpu

#
# # GPU Params:
# COMPUTE_MODE=XPU
# module load frameworks/2023-03-03-experimental
# module list
# source /home/cadams/frameworks-2023-01-31-extension/bin/activate
# export NUMEXPR_MAX_THREADS=1

python bin/exec.py \
--config-name a21-deterministic \
run.id=${run_id} \
run.distributed=False \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
run.compute_mode=${COMPUTE_MODE} \
data.data_directory=${DATA_DIR} \
mode.optimizer.loss_balance_scheme=none \
framework.inter_op_parallelism_threads=2 \
framework.intra_op_parallelism_threads=32 \
run.iterations=2500
