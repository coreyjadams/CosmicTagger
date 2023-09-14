#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q preemptable
#PBS -A datascience
#PBS -l filesystems=home:grand

# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/CosmicTagger
cd ${WORK_DIR}

DATA_DIR=/lus/grand/projects/datascience/cadams/datasets/SBND/

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=2
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo $GLOBAL_BATCH_SIZE

run_id=single-A100-fp32-GPU-4ranks
#
# # CPU params:
# export OMP_NUM_THREADS=32
# export KMP_BLOCKTIME=0
# COMPUTE_MODE=CPU
#
# GPU Params:
COMPUTE_MODE=GPU
export NVIDIA_TF32_OVERRIDE=0

module load conda
conda activate /home/cadams/miniconda3/tf-2.11.0

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=numa \
python bin/exec.py \
--config-name a21-deterministic \
run.id=${run_id} \
run.distributed=True \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
run.compute_mode=${COMPUTE_MODE} \
data.data_directory=${DATA_DIR} \
mode.optimizer.loss_balance_scheme=none \
framework.inter_op_parallelism_threads=2 \
framework.intra_op_parallelism_threads=32 \
run.iterations=2500
